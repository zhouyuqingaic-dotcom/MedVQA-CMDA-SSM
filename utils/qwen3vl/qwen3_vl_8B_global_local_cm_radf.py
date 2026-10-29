import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_utils import (
    build_biomedclip_route_features,
)
from utils.qwen3vl.qwen3_vl_8B_four_directional_ssm_utils import (
    DIRECTION_NAMES,
)


class GlobalLocalCMRADF(nn.Module):
    """
    CM-DA-SSM 第二阶段：Global–Local CM-RADF。

    Cross-Modal Response-Adaptive Directional Fusion

    本模块仅负责：
        1. 复用原 MoRA-Med 的 BiomedCLIP 2049 维跨模态先验；
        2. 生成图像—问题联合 Query；
        3. 根据四方向 SSM 响应计算 Global Direction Matching；
        4. 对每个 token 计算 Local Token Matching；
        5. 按四个扫描方向执行 token-wise Softmax；
        6. 得到 Fused Directional Features；
        7. 经 LN_fuse + Shared Up 投影生成视觉残差 R。

    本模块暂时不负责：
        - 四方向 Selective SSM（由上一步单独实现）；
        - Capped RMS Calibration；
        - lambda 残差缩放；
        - X' = X + lambda * rho * R；
        - Qwen Wrapper / DeepStack 修改。

    Input:
        directional_features:
            [1, N, 4, C]
            N 为当前图像的有效 pooled visual token 数量；
            4 个方向依次对应 RF、RR、CF、CR；
            四方向已经逆扫描回原始空间位置。

        biomed_img_feat:
            [1, 512]

        biomed_txt_feat:
            [1, 512]

    Output:
        residual:
            [1, N, D]

        若 return_details=True，同时返回供诊断的字典。
        这份字典包含计算图上的张量，记录日志时应主动 detach。

    典型参数：
        D = 4096
        C = 256
        Condition = 2049 -> 128 -> Query 64

    注意：
        该 Query 在方向融合阶段使用，不修改 SSM 的内部状态。
        本模块没有旧 MoRA-Med 的 scale_head、gate_head、Soft Gate。
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        biomedclip_image_encoder_dim: int = 512,
        biomedclip_text_encoder_dim: int = 512,
        condition_hidden_dim: int = 128,
        query_dim: int = 64,
        eta: float = 1.0,
        eta_mode: str = "fixed",
        zero_init_up: bool = True,
    ):
        super().__init__()

        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError(
                "hidden_dim 和 bottleneck_dim 必须大于 0："
                f"hidden_dim={hidden_dim}, "
                f"bottleneck_dim={bottleneck_dim}"
            )

        if (
            biomedclip_image_encoder_dim <= 0
            or biomedclip_text_encoder_dim <= 0
        ):
            raise ValueError(
                "BiomedCLIP 图文向量维度必须为正整数。"
            )

        # 原设计中的逐元素相乘、绝对差、余弦相似度
        # 均要求图像向量和文本向量具有相同维度。
        if (
            biomedclip_image_encoder_dim
            != biomedclip_text_encoder_dim
        ):
            raise ValueError(
                "CM-RADF 使用完整图文交互先验，"
                "BiomedCLIP 图像与文本维度必须一致："
                f"img={biomedclip_image_encoder_dim}, "
                f"txt={biomedclip_text_encoder_dim}"
            )

        if condition_hidden_dim <= 0 or query_dim <= 0:
            raise ValueError(
                "condition_hidden_dim 和 query_dim 必须大于 0。"
            )

        if eta_mode not in {"fixed", "learnable"}:
            raise ValueError(
                "eta_mode 必须为 'fixed' 或 'learnable'，"
                f"实际为 {eta_mode}"
            )

        if not math.isfinite(float(eta)) or eta < 0.0:
            raise ValueError(
                f"局部匹配系数 eta 必须为非负有限数，实际为 {eta}"
            )

        if eta_mode == "learnable" and eta <= 0.0:
            raise ValueError(
                "当 eta_mode='learnable' 时，"
                "eta 必须大于 0，便于进行 softplus 反初始化。"
            )

        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.biomedclip_image_encoder_dim = int(
            biomedclip_image_encoder_dim
        )
        self.biomedclip_text_encoder_dim = int(
            biomedclip_text_encoder_dim
        )
        self.condition_hidden_dim = int(
            condition_hidden_dim
        )
        self.query_dim = int(query_dim)

        self.eta_mode = eta_mode
        self.zero_init_up = bool(zero_init_up)

        # 初始化时确定是否为 Global-only 消融。
        # 避免在每个 forward 中对 GPU buffer 调用 .item()，
        # 导致不必要的 CPU/GPU 同步。
        self.global_only = (
            eta_mode == "fixed" and float(eta) == 0.0
        )

        # 与 FourDirectionalSSM 的方向输出顺序完全一致：
        # index 0 -> RF
        # index 1 -> RR
        # index 2 -> CF
        # index 3 -> CR
        self.direction_names = DIRECTION_NAMES
        self.num_directions = len(DIRECTION_NAMES)

        # ==========================================================
        # 1. BiomedCLIP Cross-Modal Prior
        # ==========================================================
        #
        # c_I, c_Q 各为 512 维，先 L2 normalize，然后拼接：
        #
        # z = [c_I;
        #      c_Q;
        #      c_I * c_Q;
        #      |c_I - c_Q|;
        #      cosine(c_I, c_Q)]
        #
        # 因此：512 * 4 + 1 = 2049。
        #
        # 具体拼接复用旧 MoRA-Med 中的：
        # build_biomedclip_route_features()
        self.condition_input_dim = (
            4 * self.biomedclip_image_encoder_dim + 1
        )

        # ==========================================================
        # 2. Condition MLP + Cross-Modal Query
        # ==========================================================
        #
        # [1, 2049] -> [1, 128] -> [1, 64]
        self.router_backbone = nn.Sequential(
            nn.Linear(
                self.condition_input_dim,
                self.condition_hidden_dim,
            ),
            nn.LayerNorm(
                self.condition_hidden_dim
            ),
            nn.GELU(),
        )

        self.query_proj = nn.Linear(
            self.condition_hidden_dim,
            self.query_dim,
        )

        # ==========================================================
        # 3. Global Direction Key
        # ==========================================================
        #
        # Y: [1, N, 4, C]
        #
        # Mean over valid tokens:
        #   [1, 4, C]
        #
        # Shared Global Key Projection:
        #   [1, 4, C] -> [1, 4, 64]
        #
        # 方向嵌入 e_d：
        #   [4, 64]，四个方向独立学习，但 Key 投影共享。
        #
        # 数学形式：
        #   k_global[d] = W_k_global * Mean(Y[d]) + e_d
        self.global_key_proj = nn.Linear(
            self.bottleneck_dim,
            self.query_dim,
            bias=False,
        )

        self.direction_embeddings = nn.Parameter(
            torch.zeros(
                self.num_directions,
                self.query_dim,
            )
        )

        # ==========================================================
        # 4. Local Token Key
        # ==========================================================
        #
        # 每个 token 与每个方向对应一个 Key：
        #
        #   [1, N, 4, C] -> [1, N, 4, 64]
        #
        # 与 Global Key 不同，这里不添加方向嵌入，
        # 仅对当前 token 的方向响应进行投影。
        self.local_key_proj = nn.Linear(
            self.bottleneck_dim,
            self.query_dim,
            bias=False,
        )

        # ==========================================================
        # 5. Global–Local Matching 系数 eta
        # ==========================================================
        #
        # score[n,d] = score_global[d] + eta * score_local[n,d]
        #
        # fixed:
        #   eta 是非负 buffer，支持 eta=0 做 Global-only 消融。
        #
        # learnable:
        #   eta = softplus(raw_eta)，训练过程中始终 > 0。
        if eta_mode == "fixed":
            self.register_buffer(
                "fixed_eta",
                torch.tensor(
                    float(eta),
                    dtype=torch.float32,
                ),
            )

            self.register_parameter(
                "raw_eta",
                None,
            )
        else:
            self.register_buffer(
                "fixed_eta",
                None,
            )

            # softplus 的反函数：
            # raw = ln(exp(eta) - 1)
            # 对较大的 eta 使用数值稳定形式。
            eta_raw_init = self._inverse_softplus(
                float(eta)
            )

            self.raw_eta = nn.Parameter(
                torch.tensor(
                    eta_raw_init,
                    dtype=torch.float32,
                )
            )

        # eta=0 时局部 Key 不参与训练。
        # 冻结分支，避免 Global-only 消融的无效参数被加入优化器。
        if eta_mode == "fixed" and eta == 0.0:
            self.local_key_proj.requires_grad_(
                False
            )

        # ==========================================================
        # 6. Shared Fusion Normalization + Shared Up
        # ==========================================================
        #
        # Fused Directional Features M：
        #   [1, N, C]
        #
        # LN_fuse + Shared Up：
        #   [1, N, C] -> [1, N, D]
        #
        # Up 权重及 bias 默认零初始化，使初始 R=0。
        self.ln_fuse = nn.LayerNorm(
            self.bottleneck_dim
        )

        self.up = nn.Linear(
            self.bottleneck_dim,
            self.hidden_dim,
        )

        if self.zero_init_up:
            nn.init.zeros_(
                self.up.weight
            )
            nn.init.zeros_(
                self.up.bias
            )

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        """
        softplus 反初始化。

        value > 0。
        对小值用 expm1，避免 exp(value)-1 的消减误差；
        对大值用渐近形式，避免 exp 溢出。
        """
        if value < 20.0:
            return math.log(
                math.expm1(value)
            )

        return (
            value
            + math.log1p(-math.exp(-value))
        )

    def _get_eta(self) -> torch.Tensor:
        """
        返回非负 eta 标量 Tensor。

        其设备由 module 参数/buffer 决定，
        分数计算时统一放到 FP32 中。
        """
        if self.eta_mode == "learnable":
            return F.softplus(
                self.raw_eta.float()
            )

        return self.fixed_eta.float()

    def _validate_inputs(
        self,
        directional_features: torch.Tensor,
        biomed_img_feat: torch.Tensor,
        biomed_txt_feat: torch.Tensor,
    ):
        """
        检查图文特征与四方向特征的形状。

        这里故意严格要求 batch=1：
        对于不同图像不同 N 的情况，上层必须先按 grid_thw 切分。

        这样 Global Mean 只对本张图片的有效 token 计算，
        不会把另一张图片的 token 或 Padding 混进全局匹配。
        """
        if (
            directional_features is None
            or directional_features.ndim != 4
        ):
            raise ValueError(
                "directional_features 必须为 [1, N, 4, C]。"
            )

        batch_size, token_count, num_directions, channels = (
            directional_features.shape
        )

        if batch_size != 1 or token_count <= 0:
            raise ValueError(
                "CM-RADF 一次只处理一个视觉条目，"
                f"实际 shape={tuple(directional_features.shape)}"
            )

        if (
            num_directions != self.num_directions
            or channels != self.bottleneck_dim
        ):
            raise ValueError(
                "四方向特征维度不匹配："
                f"actual={tuple(directional_features.shape)}, "
                f"expected=[1, N, 4, {self.bottleneck_dim}]"
            )

        if not directional_features.is_floating_point():
            raise ValueError(
                "directional_features 必须是浮点特征。"
            )

        for feat_name, feat, expected_dim in (
            (
                "biomed_img_feat",
                biomed_img_feat,
                self.biomedclip_image_encoder_dim,
            ),
            (
                "biomed_txt_feat",
                biomed_txt_feat,
                self.biomedclip_text_encoder_dim,
            ),
        ):
            if (
                feat is None
                or feat.ndim != 2
                or tuple(feat.shape) != (1, expected_dim)
            ):
                actual_shape = (
                    None
                    if feat is None
                    else tuple(feat.shape)
                )

                raise ValueError(
                    f"{feat_name} 必须为 [1, {expected_dim}]，"
                    f"实际 shape={actual_shape}"
                )

            if not feat.is_floating_point():
                raise ValueError(
                    f"{feat_name} 必须为浮点张量。"
                )

            if feat.device != directional_features.device:
                raise ValueError(
                    f"{feat_name} 与 directional_features "
                    "必须处在相同设备："
                    f"{feat.device} vs {directional_features.device}"
                )

    def _build_cross_modal_query(
        self,
        biomed_img_feat: torch.Tensor,
        biomed_txt_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造 BiomedCLIP Cross-Modal Query。

        原 MoRA-Med 已实现：
            build_biomedclip_route_features(..., True)

        这里直接复用它，避免新旧方案的条件先验口径不同。

        特征生成时先转 FP32，再使用旧函数中的 F.normalize，
        防止输入为 FP16/BF16 且数值很小时归一化不稳定。

        Returns:
            query: [1, query_dim]
        """
        route_features = build_biomedclip_route_features(
            biomed_img_feat=biomed_img_feat.float(),
            biomed_txt_feat=biomed_txt_feat.float(),
            use_cross_modal_prior=True,
        )

        if route_features.size(-1) != self.condition_input_dim:
            raise RuntimeError(
                "BiomedCLIP 条件向量维度错误："
                f"expected={self.condition_input_dim}, "
                f"actual={route_features.size(-1)}"
            )

        # 和 Condition MLP 参数保持一致的 dtype。
        # 不会 detach，条件 MLP 及 Query Projection 可训练。
        route_features = route_features.to(
            dtype=self.router_backbone[0].weight.dtype,
        )

        # [1, 2049] -> [1, 128] -> [1, 64]
        route_hidden = self.router_backbone(
            route_features
        )

        query = self.query_proj(
            route_hidden
        )

        return query

    def forward(
        self,
        directional_features: torch.Tensor,
        biomed_img_feat: torch.Tensor,
        biomed_txt_feat: torch.Tensor,
        return_details: bool = False,
    ):
        """
        Args:
            directional_features:
                4D-SSM 已经逆扫描对齐的输出。
                shape: [1, N, 4, C]

            biomed_img_feat:
                冻结 BiomedCLIP image embedding。
                shape: [1, 512]

            biomed_txt_feat:
                冻结 BiomedCLIP text embedding。
                shape: [1, 512]

            return_details:
                True 时返回 (residual, details)。
                训练主路径建议保持 False，节约诊断信息的存储。

        Returns:
            residual:
                [1, N, D]

            或：
                residual, details

        本模块不修改 pooled tokens：
            只返回 R，不计算 X + lambda * rho * R。
        """
        self._validate_inputs(
            directional_features=directional_features,
            biomed_img_feat=biomed_img_feat,
            biomed_txt_feat=biomed_txt_feat,
        )

        # 下游 Key Projection 的参数 dtype 作为计算基准。
        # 此转型保持梯度，可与不同精度的 SSM 输出衔接。
        y = directional_features.to(
            dtype=self.global_key_proj.weight.dtype,
        )

        # ==========================================================
        # Step 1. Cross-Modal Query
        # ==========================================================
        #
        # query: [1, 64]
        query = self._build_cross_modal_query(
            biomed_img_feat=biomed_img_feat,
            biomed_txt_feat=biomed_txt_feat,
        )

        # 点积匹配使用 float32，提高 BF16/FP16 时的 Softmax 稳定性。
        query_fp32 = query.float()
        score_scale = math.sqrt(
            float(self.query_dim)
        )

        # ==========================================================
        # Step 2. Global Direction Matching
        # ==========================================================
        #
        # 全局平均只在本图像 N 个有效视觉 token 上进行。
        #
        # [1, N, 4, C] -> [1, 4, C]
        global_direction_features = y.mean(
            dim=1
        )

        # [1, 4, C] -> [1, 4, 64]
        global_keys = self.global_key_proj(
            global_direction_features
        )

        # 为每个方向添加可学习方向嵌入：
        # e_RF / e_RR / e_CF / e_CR。
        #
        # [4, 64] -> [1, 4, 64]
        global_keys = (
            global_keys
            + self.direction_embeddings.to(
                dtype=global_keys.dtype
            ).unsqueeze(0)
        )

        # q: [1, 64]
        # k: [1, 4, 64]
        #
        # [1, 4]
        global_scores = (
            torch.einsum(
                "bq,bdq->bd",
                query_fp32,
                global_keys.float(),
            )
            / score_scale
        )

        # ==========================================================
        # Step 3. Local Token Matching
        # ==========================================================
        #
        # [1, N, 4, C] -> [1, N, 4, 64]
        #
        # 这里的 "Local" 指对每个视觉 token 分别匹配，
        # 没有额外加入 CNN 或局部邻域卷积。
        #
        # eta=0 时是 Global-only 消融：
        # 跳过不需要的 Local Key 计算。
        eta_value = self._get_eta()

        if self.global_only:
            local_scores = torch.zeros(
                (
                    y.size(0),
                    y.size(1),
                    self.num_directions,
                ),
                dtype=torch.float32,
                device=y.device,
            )
        else:
            local_keys = self.local_key_proj(
                y.to(
                    dtype=self.local_key_proj.weight.dtype
                )
            )

            # [1, N, 4]
            local_scores = (
                torch.einsum(
                    "bq,bndq->bnd",
                    query_fp32,
                    local_keys.float(),
                )
                / score_scale
            )

        # ==========================================================
        # Step 4. Global–Local Score + Directional Softmax
        # ==========================================================
        #
        # global_scores: [1, 4]
        # local_scores : [1, N, 4]
        #
        # 广播后：
        #   score[n,d] = global[d] + eta * local[n,d]
        #
        # scores: [1, N, 4]
        combined_scores = (
            global_scores.unsqueeze(1)
            + eta_value * local_scores
        )

        # 仅沿 4 个方向归一化！
        #
        # 不能沿 token 维 dim=1 做 Softmax。
        #
        # routing_weights: [1, N, 4]
        routing_weights = torch.softmax(
            combined_scores,
            dim=-1,
        )

        # ==========================================================
        # Step 5. Token-Wise Directional Fusion
        # ==========================================================
        #
        # Y:    [1, N, 4, C]
        # beta: [1, N, 4]
        #
        # M[n] = sum_d beta[n,d] * Y[n,d]
        #
        # M: [1, N, C]
        fused_directional_features = (
            y
            * routing_weights.to(
                dtype=y.dtype
            ).unsqueeze(-1)
        ).sum(
            dim=2
        )

        # ==========================================================
        # Step 6. LN_fuse + Shared Up Projection
        # ==========================================================
        #
        # [1, N, C] -> [1, N, D]
        #
        # 只有这一层 Up，在四方向融合之后调用一次。
        # 不在每个方向内部做四份 Up。
        fused_directional_features = (
            fused_directional_features.to(
                dtype=self.up.weight.dtype
            )
        )

        residual = self.up(
            self.ln_fuse(
                fused_directional_features
            )
        )

        # 保持与输入方向特征一致的 dtype。
        # 后续 Capped RMS 仍应在 float32 上计算统计量。
        residual = residual.to(
            dtype=directional_features.dtype
        )

        if not return_details:
            return residual

        # 供论文消融及可解释性分析使用。
        # 这些张量保留计算图，不写入 self.latest_*。
        # 如果在评估日志中保存，应在外层 detach().cpu()。
        details = {
            "query": query,
            "global_scores": global_scores,
            "local_scores": local_scores,
            "combined_scores": combined_scores,
            "routing_weights": routing_weights,
            "mean_direction_weights": routing_weights.mean(dim=1),
            "fused_directional_features": fused_directional_features,
            "eta": eta_value,
        }

        return residual, details



# =====================================================================
# A1 消融：Four-Directional SSM + Uniform Directional Fusion
# =====================================================================
class UniformFourDirectionalFusion(nn.Module):
    """
    A1 专用的均匀四方向融合模块。

    与 A2/A3 的 CM-RADF 做严格区分：
        - 只有四方向 SSM 输出 Y；
        - 四个方向的权重固定为 1/4；
        - 完全不创建 BiomedCLIP Query / Global / Local Key；
        - 不创建 direction_embeddings 或 raw_eta；
        - 保留与完整模型相同的 LN_fuse + Shared Up。

    公式：
        M[n] = (Y_RF[n] + Y_RR[n] + Y_CF[n] + Y_CR[n]) / 4
        R[n] = SharedUp(LN_fuse(M[n]))

    因为均匀权重是常量，本模块既不需要 BiomedCLIP 特征，
    也不需要加载冻结的 BiomedCLIP 编码器。

    初始化 Shared Up=0 时，R=0，下游残差注入仍然严格恒等。
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        zero_init_up: bool = True,
    ):
        super().__init__()

        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError(
                "hidden_dim 和 bottleneck_dim 必须为正数。"
            )

        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_directions = 4
        self.direction_names = DIRECTION_NAMES

        # 四方向完成融合后，仅执行一份 LN_fuse 和 Shared Up。
        self.ln_fuse = nn.LayerNorm(
            self.bottleneck_dim
        )

        self.up = nn.Linear(
            self.bottleneck_dim,
            self.hidden_dim,
        )

        if zero_init_up:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(
        self,
        directional_features: torch.Tensor,
        biomed_img_feat: torch.Tensor | None = None,
        biomed_txt_feat: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        """
        Args:
            directional_features: [1,N,4,C]，四方向均已经逆扫描。
            biomed_img_feat/txt_feat: 接口兼容占位，A1 永不使用。
            return_details: 是否提供每个 token 的均匀融合权重。

        Returns:
            R: [1,N,D]（未注入的视觉残差）。
        """
        if (
            directional_features is None
            or directional_features.ndim != 4
            or directional_features.size(0) != 1
            or directional_features.size(1) <= 0
            or directional_features.size(2) != self.num_directions
            or directional_features.size(3) != self.bottleneck_dim
        ):
            raise ValueError(
                "A1 的 directional_features 必须是 "
                f"[1,N,4,{self.bottleneck_dim}]。"
            )

        # 使用原始顺序计算四方向均值；不引入可学习路由。
        # [1,N,4,C] -> [1,N,C]
        fused = directional_features.mean(dim=2)

        # 按 Shared Up 的参数精度执行 LN + Up。
        residual = self.up(
            self.ln_fuse(
                fused.to(dtype=self.up.weight.dtype)
            )
        ).to(dtype=directional_features.dtype)

        if not return_details:
            return residual

        # 诊断 beta[n,d] 恒为 0.25；此 Tensor 不需要计算梯度。
        weights = torch.full(
            (
                directional_features.size(0),
                directional_features.size(1),
                4,
            ),
            0.25,
            dtype=torch.float32,
            device=directional_features.device,
        )

        details = {
            "routing_weights": weights,
            "mean_direction_weights": weights.mean(dim=1),
            "fused_directional_features": fused,
            "fusion_mode": "uniform",
        }

        return residual, details
