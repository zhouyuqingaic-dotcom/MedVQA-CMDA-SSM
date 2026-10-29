import math
from typing import Optional

import torch
import torch.nn as nn

from utils.qwen3vl.qwen3_vl_8B_four_directional_cm_radf import (
    Qwen3VLFourDirectionalCMRADF,
)
from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_utils import (
    safe_logit,
    visual_token_split,
)


# =====================================================================
# CM-DA-SSM Visual Residual Adapter
#
# 计算流程：
#   Pooled Visual Tokens X
#       -> Four-Directional SSM
#       -> Global–Local CM-RADF
#       -> Shared Up
#       -> Visual Residual R
#       -> Capped RMS Calibration
#       -> Learnable Lambda
#       -> Enhanced Pooled Tokens X'
#
# 约束：
#   1. 本模块只接收并返回 pooled visual tokens；
#   2. DeepStack 不传进来，也不在这里修改；
#   3. 不再使用旧 MoRA-Med 的 Scale Router 或 Soft Gate；
#   4. 每个视觉条目分别计算 RMS，不跨图像混合；
#   5. Shared Up 零初始化时 R=0，因此 X'=X（精确恒等）。
# =====================================================================
class Qwen3VLCMDASSMVisualAdapter(nn.Module):
    """
    CM-DA-SSM pooled-only 视觉残差适配器，支持四组视觉分支消融。

    与原 MoRA-Med 的 Adapter 前向输入风格保持一致：

        forward(
            x,
            biomed_img_feat,
            biomed_txt_feat,
            grid_thw,
        )

    Args:
        hidden_dim:
            Qwen3-VL pooled visual token 通道数，通常为 4096。

        r:
            共享降维比，C = hidden_dim // r。
            hidden_dim=4096、r=16 时 C=256。

        ssm_d_state / ssm_d_conv / ssm_expand:
            四方向 Selective SSM 的内部超参数。
            默认值是初始参考设置，仍需正式实验验证。

        ssm_backend:
            "mamba"：正式 CUDA 训练；
            "torch_reference"：CPU/单元测试，权重与 Mamba 不兼容。

        eta / eta_mode:
            Global–Local CM-RADF 的局部匹配贡献系数。

        lambda_init:
            固定模式的常量或可学习模式的初始值，必须位于 (0,1)。
            当前新架构设为 0.2。

        residual_norm_eps:
            RMS 计算的数值稳定常数。

        residual_norm_ratio_clip:
            RMS 比率上限 rho_max，必须为正有限数。
            默认设置为 10.0。

        zero_init_up:
            是否将 Shared Up 的 weight/bias 设为 0。
            正式新模型建议保持 True。

    Input:
        x: [N_total, D] 或 [1, N_total, D]
        biomed_img_feat: [num_visual_items, 512]
        biomed_txt_feat: [num_visual_items, 512]
        grid_thw: [num_visual_items, 3]，Qwen merger 前的网格。

    Output:
        enhanced_x:
            与输入 x 完全相同的 shape 和 dtype。

    特别注意：
        本类的输出是已经注入残差的 X'，
        与上一步组合模块只输出 R 的接口不同。
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        r: int = 16,

        # -------------------------------------------------------------
        # Four-Directional SSM
        # -------------------------------------------------------------
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        ssm_backend: str = "mamba",

        # -------------------------------------------------------------
        # BiomedCLIP + Global–Local CM-RADF
        # -------------------------------------------------------------
        biomedclip_image_encoder_dim: int = 512,
        biomedclip_text_encoder_dim: int = 512,
        condition_hidden_dim: int = 128,
        query_dim: int = 64,
        eta: float = 1.0,
        eta_mode: str = "fixed",

        # -------------------------------------------------------------
        # Capped RMS + Learnable Lambda
        # -------------------------------------------------------------
        lambda_init: float = 0.2,
        residual_norm_eps: float = 1e-6,
        residual_norm_ratio_clip: float = 10.0,
        zero_init_up: bool = True,
        fusion_mode: str = "cm_radf",
        lambda_mode: str = "learnable",
        use_rms_norm: bool = True,
    ):
        super().__init__()

        # 五组消融的计算开关：
        # A1 = uniform + fixed lambda + Capped RMS
        # A2 = cm_radf + fixed lambda + Capped RMS
        # A3 = cm_radf + learnable lambda + Capped RMS
        # A3-NoRMS = cm_radf + learnable lambda + rho=1
        #
        # A0 的 LoRA-only 不实例化本类，直接在 Wrapper 跳过。
        if fusion_mode not in {"uniform", "cm_radf"}:
            raise ValueError(
                f"fusion_mode 非法：{fusion_mode!r}"
            )
        if lambda_mode not in {"fixed", "learnable"}:
            raise ValueError(
                f"lambda_mode 必须是 fixed/learnable，实际为 {lambda_mode!r}"
            )

        self.fusion_mode = fusion_mode
        self.lambda_mode = lambda_mode
        self.use_rms_norm = bool(use_rms_norm)

        if not math.isfinite(float(lambda_init)) or not (0.0 < lambda_init < 1.0):
            raise ValueError(
                "lambda_init 必须在 (0, 1) 之间，"
                f"当前为 {lambda_init}"
            )

        if (
            not math.isfinite(float(residual_norm_eps))
            or residual_norm_eps <= 0.0
        ):
            raise ValueError(
                "residual_norm_eps 必须是正有限数，"
                f"当前为 {residual_norm_eps}"
            )

        if (
            not math.isfinite(float(residual_norm_ratio_clip))
            or residual_norm_ratio_clip <= 0.0
        ):
            raise ValueError(
                "residual_norm_ratio_clip / rho_max "
                "必须是正有限数，"
                f"当前为 {residual_norm_ratio_clip}"
            )

        self.hidden_dim = int(hidden_dim)
        self.residual_norm_eps = float(residual_norm_eps)
        self.residual_norm_ratio_clip = float(residual_norm_ratio_clip)

        # =============================================================
        # 1. Four-Directional SSM + Global–Local CM-RADF
        # =============================================================
        #
        # 直接复用 Step 2 的组合类，不复制 SSM / Fusion 逻辑。
        #
        # 其输出是未注入的残差 R：
        #   [N_total,D] -> [N_total,D]
        self.feature_adapter = Qwen3VLFourDirectionalCMRADF(
            hidden_dim=hidden_dim,
            r=r,
            ssm_d_state=ssm_d_state,
            ssm_d_conv=ssm_d_conv,
            ssm_expand=ssm_expand,
            ssm_backend=ssm_backend,

            biomedclip_image_encoder_dim=(
                biomedclip_image_encoder_dim
            ),
            biomedclip_text_encoder_dim=(
                biomedclip_text_encoder_dim
            ),
            condition_hidden_dim=condition_hidden_dim,
            query_dim=query_dim,
            eta=eta,
            eta_mode=eta_mode,
            zero_init_up=zero_init_up,
            fusion_mode=fusion_mode,
        )

        # =============================================================
        # 2. Learnable Lambda
        # =============================================================
        #
        # λ = sigmoid(a)
        #
        # 没有额外的 lambda_max、独立 Gate 或逐样本 Gate。
        # λ 为全局共享的一个标量参数。
        #
        # 初始化：
        #   a(0) = log(0.2 / 0.8)
        #   sigmoid(a(0)) = 0.2
        #
        # 保持 lambda_a 为 float32，避免 BF16 参数量化影响初值。
        if self.lambda_mode == "learnable":
            # A3 / A3-NoRMS：参数将加入 optimizer.lambda 组。
            self.lambda_a = nn.Parameter(
                torch.tensor(
                    safe_logit(float(lambda_init)),
                    dtype=torch.float32,
                )
            )
            self.register_buffer("fixed_lambda", None)
        else:
            # A1 / A2：真实固定 λ=0.2，不使用 requires_grad=False
            # 的伪参数，确保优化器和 checkpoint 都能明确识别。
            self.register_parameter("lambda_a", None)
            self.register_buffer(
                "fixed_lambda",
                torch.tensor(
                    float(lambda_init),
                    dtype=torch.float32,
                ),
            )

    def _compute_lambda(self) -> torch.Tensor:
        """
        计算固定或可训练的全局缩放系数。

        Returns:
            lambda_value: scalar FP32 Tensor, 0 < λ < 1。

        不执行 detach，保证非零残差出现后 lambda_a 可训练。
        """
        if self.lambda_mode == "fixed":
            return self.fixed_lambda.float()

        return torch.sigmoid(
            self.lambda_a.float()
        )

    def _calibrate_one_image(
        self,
        x_i: torch.Tensor,
        residual_i: torch.Tensor,
        lambda_value: torch.Tensor,
        return_stats: bool = False,
    ):
        """
        对单张图像 / 视频条目的全部有效 pooled tokens 计算 RMS。

        Args:
            x_i:
                [1, N_i, D]

            residual_i:
                [1, N_i, D]

            lambda_value:
                共享 λ 标量。

            return_stats:
                True 时才计算额外的注入 RMS 等诊断量。
                常规训练时 False，避免保留不需要的计算图。

        Returns:
            enhanced_i:
                [1, N_i, D]，与 x_i dtype 相同。

            stats:
                FP32 的 RMS / rho / 注入幅度诊断信息。

        数学定义：
            RMS(A) = sqrt(mean(A**2) + eps)

            rho = min(
                RMS(X_i) / RMS(R_i),
                rho_max,
            )

            delta_X_i = lambda * rho * R_i
            X'_i = X_i + delta_X_i

        关键实现细节：
            - RMS 在 FP32 上算，避免 FP16/BF16 溢出。
            - rho 按当前图像独立计算，不跨图像求均值。
            - eps 避免 R_i=0 时除零。
            - 不对 rho / lambda 使用 detach()，保证可微。
            - 只在最终注入时将 delta 转回 pooled token dtype。
        """
        if x_i.shape != residual_i.shape:
            raise ValueError(
                "单图 pooled tokens 与残差形状必须相同："
                f"x_i={tuple(x_i.shape)}, "
                f"residual_i={tuple(residual_i.shape)}"
            )

        x_fp32 = x_i.float()
        residual_fp32 = residual_i.float()

        if self.use_rms_norm or return_stats:
            # Capped RMS：按单张图片所有有效 pooled tokens × D
            # 求均方根；FP32 计算避免混合精度溢出。
            rms_x = torch.sqrt(
                x_fp32.square().mean(
                    dim=(-2, -1),
                    keepdim=True,
                )
                + self.residual_norm_eps
            )

            rms_r = torch.sqrt(
                residual_fp32.square().mean(
                    dim=(-2, -1),
                    keepdim=True,
                )
                + self.residual_norm_eps
            )
        else:
            # A3-NoRMS 在普通训练前向中完全跳过 RMS 计算。
            rms_x = None
            rms_r = None

        if self.use_rms_norm:
            # A1/A2/A3：Capped RMS。
            # 对每张图独立计算，ρ 上限为 residual_norm_ratio_clip。
            rho = torch.clamp(
                rms_x / rms_r,
                max=self.residual_norm_ratio_clip,
            )
        else:
            # A3-NoRMS：严格恒等缩放 ρ=1。
            #
            # 不是将 rho_max 改大，不是 detach(rho)，
            # 也不是推理阶段关闭 RMS。
            # 在 Stage 1 和 Stage 2 的每次 forward 都只使用 1。
            rho = torch.ones(
                (x_i.size(0), 1, 1),
                device=x_i.device,
                dtype=torch.float32,
            )

        # 显式 FP32 乘法，最后转换回 Qwen pooled token dtype。
        delta_fp32 = (
            lambda_value.float()
            * rho
            * residual_fp32
        )

        delta_x = delta_fp32.to(
            device=x_i.device,
            dtype=x_i.dtype,
        )

        # Shared Up 为 0 时 residual_i 精确为 0，
        # delta_x 也精确为 0，因此 x_i' == x_i。
        enhanced_i = x_i + delta_x

        if not return_stats:
            return enhanced_i, None

        # 诊断数据仍为当前计算图的一部分；
        # 调用者做日志或保存时应 detach().cpu()。
        #
        # delta_rms 是真实注入幅度：当 R=0 时必须等于 0，
        # 因此这里不额外添加 eps。
        stats = {
            "rms_x": rms_x.squeeze(),
            "rms_r": rms_r.squeeze(),
            "rho": rho.squeeze(),
            "delta_rms": torch.sqrt(
                delta_fp32.square().mean(
                    dim=(-2, -1),
                    keepdim=True,
                )
            ).squeeze(),
        }

        return enhanced_i, stats

    def forward(
        self,
        x: torch.Tensor,
        biomed_img_feat: torch.Tensor,
        biomed_txt_feat: torch.Tensor,
        grid_thw: torch.Tensor,
        return_details: bool = False,
    ):
        """
        Args:
            x:
                Qwen3-VL pooled visual tokens：
                [N_total,D] 或 [1,N_total,D]。

            biomed_img_feat:
                冻结 BiomedCLIP image embedding，逐视觉条目对齐。

            biomed_txt_feat:
                冻结 BiomedCLIP text embedding，逐视觉条目对齐。

            grid_thw:
                Qwen merger 前的 grid，形状 [num_visual_items,3]。

            return_details:
                True 返回 X' 和逐图的融合权重 / RMS 统计；
                False 只返回 X'，降低训练中的诊断内存开销。

        Returns:
            enhanced_x:
                与 x 完全相同 shape 和 dtype 的增强 pooled tokens。

            可选：
                enhanced_x, details_per_image

        这里从始至终不读取 / 不修改 DeepStack features。
        """
        if x is None or x.ndim not in (2, 3):
            raise ValueError(
                "pooled tokens x 必须为 [N_total,D] "
                "或 [1,N_total,D]。"
            )

        input_was_2d = (x.ndim == 2)
        x_3d = x.unsqueeze(0) if input_was_2d else x

        if (
            x_3d.size(0) != 1
            or x_3d.size(-1) != self.hidden_dim
        ):
            raise ValueError(
                "pooled tokens 输入形状错误："
                f"{tuple(x.shape)}，"
                f"期望通道 D={self.hidden_dim}"
            )

        # =============================================================
        # Step 1. 生成 Shared Up 后的视觉残差 R
        # =============================================================
        #
        # Feature Adapter 内部：
        #   visual_token_split
        #       -> 4D-SSM
        #       -> Global–Local CM-RADF
        #       -> Shared Up
        #
        # R 的形状与 x 完全一致，内容尚未注入。
        if return_details:
            residual, details_per_image = self.feature_adapter(
                x=x,
                biomed_img_feat=biomed_img_feat,
                biomed_txt_feat=biomed_txt_feat,
                grid_thw=grid_thw,
                return_details=True,
            )
        else:
            residual = self.feature_adapter(
                x=x,
                biomed_img_feat=biomed_img_feat,
                biomed_txt_feat=biomed_txt_feat,
                grid_thw=grid_thw,
                return_details=False,
            )
            details_per_image = None

        residual_3d = (
            residual.unsqueeze(0)
            if input_was_2d
            else residual
        )

        if residual_3d.shape != x_3d.shape:
            raise RuntimeError(
                "视觉残差 R 与 pooled tokens 形状不一致："
                f"X={tuple(x_3d.shape)}, "
                f"R={tuple(residual_3d.shape)}"
            )

        # =============================================================
        # Step 2. 根据 grid_thw 对 X 与 R 作相同的图像级切分
        # =============================================================
        #
        # 注意 RMS 不能在整个 batch 的所有视觉 token 上求均值。
        # 每个图像 / 视频条目应该拥有自己的 rho_i。
        #
        # 这里复用旧工具的 spatial_merge_size=2 假设。
        # 正式适配其他视觉塔配置时，需要同步修改旧工具。
        split_x = visual_token_split(
            x=x_3d,
            grid_thw=grid_thw,
        )

        split_r = visual_token_split(
            x=residual_3d,
            grid_thw=grid_thw,
        )

        if len(split_x) != len(split_r):
            raise RuntimeError(
                "pooled tokens 与残差切分后的图像数量不一致。"
            )

        # =============================================================
        # Step 3. Learnable Lambda
        # =============================================================
        #
        # 对所有图片共用同一个可学习 λ。
        lambda_value = self._compute_lambda()

        # =============================================================
        # Step 4. 逐图 Capped RMS Calibration + Residual Injection
        # =============================================================
        enhanced_parts = []

        for image_index, (
            (x_i, grid_shape_x),
            (residual_i, grid_shape_r),
        ) in enumerate(zip(split_x, split_r)):
            if grid_shape_x != grid_shape_r:
                raise RuntimeError(
                    f"第 {image_index} 张图片的网格与残差不对齐："
                    f"X={grid_shape_x}, R={grid_shape_r}"
                )

            enhanced_i, stats_i = self._calibrate_one_image(
                x_i=x_i,
                residual_i=residual_i,
                lambda_value=lambda_value,
                return_stats=return_details,
            )

            enhanced_parts.append(
                enhanced_i
            )

            if return_details:
                details_per_image[image_index].update(
                    stats_i
                )
                details_per_image[image_index]["lambda"] = (
                    lambda_value
                )

        # =============================================================
        # Step 5. 恢复原视觉 token 顺序与形状
        # =============================================================
        #
        # [1,N_0,D] + [1,N_1,D] + ...
        #       -> [1,N_total,D]
        enhanced_x = torch.cat(
            enhanced_parts,
            dim=1,
        )

        if enhanced_x.shape != x_3d.shape:
            raise RuntimeError(
                "增强 pooled tokens 的形状发生变化："
                f"input={tuple(x_3d.shape)}, "
                f"output={tuple(enhanced_x.shape)}"
            )

        if enhanced_x.dtype != x_3d.dtype:
            raise RuntimeError(
                "增强 pooled tokens 的 dtype 发生变化："
                f"input={x_3d.dtype}, output={enhanced_x.dtype}"
            )

        if input_was_2d:
            enhanced_x = enhanced_x.squeeze(0)

        if return_details:
            return enhanced_x, details_per_image

        return enhanced_x
