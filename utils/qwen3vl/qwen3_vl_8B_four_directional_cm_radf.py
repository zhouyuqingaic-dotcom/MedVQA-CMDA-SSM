import torch
import torch.nn as nn

from utils.qwen3vl.qwen3_vl_8B_four_directional_ssm import (
    FourDirectionalSSM,
)
from utils.qwen3vl.qwen3_vl_8B_global_local_cm_radf import (
    GlobalLocalCMRADF,
    UniformFourDirectionalFusion,
)
from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_utils import (
    visual_token_split,
)


class Qwen3VLFourDirectionalCMRADF(nn.Module):
    """
    Four-Directional SSM + Global–Local CM-RADF 的独立组合模块。

    设计目的：
        让 CM-DA-SSM 的前两个核心阶段可以直接对接旧 MoRA-Med
        的视觉输出与 BiomedCLIP 图文特征，同时不提前引入
        Capped RMS / lambda / DeepStack 等下一阶段逻辑。

    与旧版 Qwen3VLMoEVisualAdapterFusion 的接口对应：

        x:
            [N_total, D] 或 [1, N_total, D]

        biomed_img_feat:
            [num_visual_items, 512]

        biomed_txt_feat:
            [num_visual_items, 512]

        grid_thw:
            [num_visual_items, 3]
            是 Qwen 原始视觉网格，而不是 spatial merge 后的网格。

    计算：
        1. 复用旧 visual_token_split，按图像切分 pooled tokens；
        2. 对每张图片执行共享降维、4D-SSM、逆扫描；
        3. 用本图像对应的 BiomedCLIP 图文向量运行 CM-RADF；
        4. LN_fuse + Shared Up，生成 [1, N_i, D] 残差；
        5. 依原顺序拼接所有图片的残差。

    输出：
        residual: 与 x 同形状的残差 R，而非增强后的 X'。

    注意：
        当前不是可直接替换旧 Wrapper 的完整 Adapter！

        旧 Wrapper 期望 self.res_adapter(...) 返回 X'。
        本类只返回 R，后续必须再执行 Capped RMS 与
        X' = X + lambda * rho * R。

        这一步刻意不修改原 Qwen Wrapper 和 DeepStack。
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        r: int = 16,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        ssm_backend: str = "mamba",
        biomedclip_image_encoder_dim: int = 512,
        biomedclip_text_encoder_dim: int = 512,
        condition_hidden_dim: int = 128,
        query_dim: int = 64,
        eta: float = 1.0,
        eta_mode: str = "fixed",
        zero_init_up: bool = True,
        fusion_mode: str = "cm_radf",
    ):
        super().__init__()

        # A1：四方向等权融合；A2/A3：完整 Global–Local CM-RADF。
        if fusion_mode not in {"uniform", "cm_radf"}:
            raise ValueError(
                f"fusion_mode 必须为 uniform/cm_radf，实际为 {fusion_mode!r}。"
            )
        self.fusion_mode = fusion_mode

        # ==========================================================
        # 1. Four-Directional Selective SSM
        # ==========================================================
        #
        # [1, N_i, D] -> [1, N_i, 4, C]
        self.four_directional_ssm = FourDirectionalSSM(
            hidden_dim=hidden_dim,
            r=r,
            ssm_d_state=ssm_d_state,
            ssm_d_conv=ssm_d_conv,
            ssm_expand=ssm_expand,
            ssm_backend=ssm_backend,
        )

        bottleneck_dim = (
            self.four_directional_ssm.bottleneck_dim
        )

        # ==========================================================
        # 2. Global–Local CM-RADF
        # ==========================================================
        #
        # [1, N_i, 4, C] + BiomedCLIP Image/Text
        #                 -> [1, N_i, D] Residual
        if self.fusion_mode == "uniform":
            # A1：只留下 LN_fuse 和 Shared Up。
            # 不构建 Query/Key/Direction Embeddings 等路由参数。
            self.cm_radf = UniformFourDirectionalFusion(
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                zero_init_up=zero_init_up,
            )
        else:
            # A2 / A3 / A3-NoRMS：完整 Global–Local CM-RADF。
            self.cm_radf = GlobalLocalCMRADF(
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
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
            )

        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)

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
                Qwen pooled visual tokens。
                [N_total, D] 或 [1, N_total, D]

            biomed_img_feat:
                每个视觉条目对应的 BiomedCLIP 图像特征。
                [num_visual_items, 512]

            biomed_txt_feat:
                每个视觉条目对应的 BiomedCLIP 条件文本特征。
                [num_visual_items, 512]

            grid_thw:
                Qwen merger 前的原始 grid。
                [num_visual_items, 3]

            return_details:
                True 时返回 (residual, diagnostics_per_item)。

        Returns:
            residual:
                与 x 的 shape 相同，尚未注入到 x。

            diagnostics_per_item:
                列表，长度为视觉条目数；
                每项包含该图像的 Global / Local scores、
                token-wise routing_weights 等信息。
        """
        if x is None or x.ndim not in {2, 3}:
            raise ValueError(
                "x 必须为 [N_total, D] 或 [1, N_total, D]。"
            )

        is_2d = (x.ndim == 2)

        if is_2d:
            x = x.unsqueeze(0)

        if x.size(0) != 1 or x.size(-1) != self.hidden_dim:
            raise ValueError(
                "pooled visual tokens 形状不符合要求："
                f"actual={tuple(x.shape)}, "
                f"expected=[1, N_total, {self.hidden_dim}]"
            )

        if grid_thw is None:
            raise ValueError(
                "必须提供用于空间对齐的 grid_thw。"
            )

        # A1 无跨模态 Query，因此无需 BioMedCLIP。
        # A2/A3 仍要求每张图像有一对图文条件向量。
        if self.fusion_mode == "cm_radf":
            if biomed_img_feat is None or biomed_txt_feat is None:
                raise ValueError(
                    "CM-RADF 融合必须提供 BioMedCLIP 图像/问题特征。"
                )

            if (
                biomed_img_feat.ndim != 2
                or biomed_txt_feat.ndim != 2
            ):
                raise ValueError(
                    "BioMedCLIP 图文特征必须是 [num_visual_items,512]。"
                )

        if (
            grid_thw.ndim != 2
            or grid_thw.size(-1) != 3
        ):
            raise ValueError(
                "grid_thw 必须为 [num_visual_items, 3]，"
                f"实际为 {tuple(grid_thw.shape)}"
            )

        # 使用原 MoRA-Med 的有效 token 切分规则，
        # 不将不同图像拼成一条连续的 SSM 序列。
        #
        # splits:
        #   [(sub_x0, (t0, H0, W0)), ...]
        #
        # H、W 在旧工具内根据 spatial_merge_size=2 转换。
        splits = visual_token_split(
            x=x,
            grid_thw=grid_thw,
        )

        num_visual_items = len(splits)

        if self.fusion_mode == "cm_radf" and (
            biomed_img_feat.size(0) != num_visual_items
            or biomed_txt_feat.size(0) != num_visual_items
        ):
            raise ValueError(
                "视觉条目数量与 BiomedCLIP 图文特征数量不一致："
                f"num_visual_items={num_visual_items}, "
                f"img_batch={biomed_img_feat.size(0)}, "
                f"txt_batch={biomed_txt_feat.size(0)}"
            )

        if self.fusion_mode == "cm_radf" and (
            biomed_img_feat.device != x.device
            or biomed_txt_feat.device != x.device
        ):
            raise ValueError(
                "pooled tokens、BiomedCLIP 图像与文本特征 "
                "必须位于同一设备。"
            )

        residual_list = []
        details_list = []

        for image_index, (
            sub_x,
            grid_shape,
        ) in enumerate(splits):
            # ======================================================
            # 1. 该张图片 / 视频条目的 Four-Directional SSM
            # ======================================================
            #
            # [1, N_i, D] -> [1, N_i, 4, C]
            #
            # 模块只针对单张图片的有效 token 工作，
            # video 的不同帧由 4D-SSM 在 batch 维中处理。
            sub_x_ssm = sub_x.to(
                dtype=self.four_directional_ssm.down.weight.dtype
            )

            directional_features = self.four_directional_ssm(
                x=sub_x_ssm,
                grid_shape=grid_shape,
            )

            # ======================================================
            # 2. 选取当前视觉条目对应的 BiomedCLIP 条件
            # ======================================================
            #
            # 使用 [i:i+1] 保留 batch 维：
            #   [1, 512] / [1, 512]
            #
            # 不允许把 B 张图片的条件平均后再喂给该图像。
            if self.fusion_mode == "uniform":
                # A1：不读取图文条件，确保完全均匀融合。
                current_img_feat = None
                current_txt_feat = None
            else:
                current_img_feat = biomed_img_feat[
                    image_index:image_index + 1
                ]
                current_txt_feat = biomed_txt_feat[
                    image_index:image_index + 1
                ]

            # ======================================================
            # 3. Global–Local CM-RADF + Shared Up
            # ======================================================
            #
            # [1, N_i, 4, C]
            #    + [1, 512] / [1, 512]
            #        -> [1, N_i, D]
            #
            # 当前只返回 R，不注入 pooled tokens。
            if return_details:
                sub_residual, sub_details = self.cm_radf(
                    directional_features=directional_features,
                    biomed_img_feat=current_img_feat,
                    biomed_txt_feat=current_txt_feat,
                    return_details=True,
                )

                # 记录该诊断信息对应的视觉条目。
                # 不对其中的 Tensor 做 detach，
                # 由调用端决定是否保留梯度或转到 CPU。
                sub_details["visual_item_index"] = image_index
                sub_details["grid_shape"] = grid_shape

                details_list.append(
                    sub_details
                )
            else:
                sub_residual = self.cm_radf(
                    directional_features=directional_features,
                    biomed_img_feat=current_img_feat,
                    biomed_txt_feat=current_txt_feat,
                    return_details=False,
                )

            # 返回给后续残差注入步骤时，
            # 与原 pooled visual tokens 的 dtype 保持一致。
            sub_residual = sub_residual.to(
                dtype=sub_x.dtype
            )

            residual_list.append(
                sub_residual
            )

        # 所有残差依照 Qwen 原先的 visual token 顺序合并。
        #
        # [1, N_0, D], [1, N_1, D], ...
        #      -> [1, N_total, D]
        residual = torch.cat(
            residual_list,
            dim=1,
        )

        if residual.shape != x.shape:
            raise RuntimeError(
                "CM-RADF 残差输出与 pooled tokens 不对齐："
                f"input={tuple(x.shape)}, "
                f"residual={tuple(residual.shape)}"
            )

        # 恢复旧 Wrapper 原有的 [N_total, D] 输入接口。
        if is_2d:
            residual = residual.squeeze(0)

        if return_details:
            return residual, details_list

        return residual
