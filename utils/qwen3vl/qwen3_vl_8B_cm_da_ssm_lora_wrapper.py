import types
from typing import List, Optional

import torch
import torch.nn as nn

from utils.ddp.ddp_utils import ddp_print
from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_visual_adapter import (
    Qwen3VLCMDASSMVisualAdapter,
)


# =====================================================================
# 冻结的 BioMedCLIP 跨模态特征提取器（CM-DA-SSM 版本）
# =====================================================================
class FrozenBioMedCLIPFeatureExtractor(nn.Module):
    """
    BioMedCLIP Frozen Image/Text Feature Extractor。

    保持与原 MoRA-Med loader 返回的接口一致：
        forward(image_tensors, text_tokens) -> (img_feat, txt_feat)

    约束：
        - 不更新 BiomedCLIP 预训练权重；
        - 即使外层 Trainer 调用 model.train()，仍然保持 eval；
        - 不允许图文特征回传梯度到 BiomedCLIP。
    """

    def __init__(self, raw_model: nn.Module):
        super().__init__()

        self.model = raw_model
        self.model.eval()
        self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        """
        覆盖 nn.Module.train()，强制冻结的 BioMedCLIP 保持 eval。

        原 MoRA-Med loader 目前导入旧版 FrozenBioMedCLIPFeatureExtractor；
        若不修改 loader，下面的 Wrapper 也会在每次提取时显式 eval，
        因此不强制要求先修改旧 loader。
        """
        super().train(False)
        self.model.eval()
        return self

    def forward(
        self,
        biomed_img_tensors: torch.Tensor,
        biomed_txt_tokens: torch.Tensor,
    ):
        with torch.no_grad():
            self.model.eval()

            img_feat, txt_feat, _ = self.model(
                biomed_img_tensors,
                biomed_txt_tokens,
            )

            img_feat = torch.nn.functional.normalize(
                img_feat,
                dim=-1,
            )

            txt_feat = torch.nn.functional.normalize(
                txt_feat,
                dim=-1,
            )

        return img_feat, txt_feat


# =====================================================================
# 视觉塔 forward patch：共享 CM-DA-SSM 同时增强 pooled + DeepStack
# =====================================================================
def install_cm_da_ssm_vision_patch(
    vision_tower: nn.Module,
    visual_adapter: Qwen3VLCMDASSMVisualAdapter,
):
    """在 Qwen3-VL Vision Merger 后对全部四条视觉流共享同一个 Adapter。

    常规 Qwen3-VL：pooler_output + 三条 deepstack_features；
    均为 Merger 后的 [N, D] 视觉 token。保留原视觉返回类型，
    让 LLM 沿原生路径在输入处使用 pooled，在前几个 Decoder 层注入 DeepStack。

    A1 为 Uniform Fusion（不需 BioMedCLIP）；A2/A3/A3-NoRMS 共用
    当前 batch 的 BioMedCLIP 条件。四次调用同一模块，不复制参数。

    目前 SSM 网格切分基于 spatial_merge_size=2，遇到其它配置直接报错。
    """
    if getattr(vision_tower, "_cm_da_ssm_vision_patched", False):
        raise RuntimeError("当前 Vision Tower 已安装 CM-DA-SSM patch，禁止重复挂载。")
    if hasattr(vision_tower, "res_adapter"):
        raise RuntimeError(
            "Vision Tower 已存在 res_adapter，请新建 Qwen3-VL 模型实例，"
            "避免覆盖原有视觉模块。"
        )
    if not isinstance(visual_adapter, nn.Module):
        raise TypeError("visual_adapter 必须是 nn.Module。")

    # 一份 Adapter 参数，训练与保存仍使用原有 visual.res_adapter 路径。
    vision_tower.res_adapter = visual_adapter
    vision_tower.original_forward = vision_tower.forward
    vision_tower.current_biomed_img_feat = None
    vision_tower.current_biomed_txt_feat = None

    def patched_vision_forward(self, *args, **kwargs):
        outputs = self.original_forward(*args, **kwargs)
        img_f = getattr(self, "current_biomed_img_feat", None)
        txt_f = getattr(self, "current_biomed_txt_feat", None)

        if (
            self.res_adapter.fusion_mode == "cm_radf"
            and (img_f is None or txt_f is None)
        ):
            raise ValueError(
                "CM-RADF 当前前向缺少 BioMedCLIP 图像/文本特征；"
                "训练请传 biomed_image_tensors/biomed_text_tokens，"
                "生成时应先为 Vision Tower 缓存两组特征。"
            )

        grid_thw = kwargs.get("grid_thw")
        if grid_thw is None and len(args) > 1:
            grid_thw = args[1]
        if grid_thw is None:
            raise ValueError("CM-DA-SSM 视觉前向没有 grid_thw。")

        merge_size = getattr(self, "spatial_merge_size", None)
        if merge_size is None:
            merge_size = getattr(getattr(self, "config", None), "spatial_merge_size", 2)
        if merge_size != 2:
            raise RuntimeError(
                f"当前 4D-SSM grid split 仅支持 spatial_merge_size=2，实际={merge_size}。"
                "请先同步修改 visual_token_split，不能静默使用错误空间网格。"
            )

        if hasattr(outputs, "pooler_output"):
            output_format = "model_output"
            pooled = outputs.pooler_output
            deepstack = getattr(outputs, "deepstack_features", None)
        elif isinstance(outputs, (tuple, list)):
            if len(outputs) != 2:
                raise RuntimeError("Qwen Vision tuple/list 输出必须为 (pooled, deepstack)。")
            output_format = "tuple" if isinstance(outputs, tuple) else "list"
            pooled, deepstack = outputs
        else:
            raise RuntimeError(
                "当前 Vision Tower 返回类型不支持："
                f"{type(outputs).__name__}。"
            )

        if not isinstance(pooled, torch.Tensor) or pooled.ndim not in (2, 3):
            raise RuntimeError("pooler_output 必须为 [N,D] 或 [1,N,D] Tensor。")
        if not isinstance(deepstack, (list, tuple)) or not deepstack:
            raise RuntimeError(
                "必须提供非空 deepstack_features 列表，"
                "本实验不允许跳过 DeepStack 适配。"
            )

        indexes = getattr(self, "deepstack_visual_indexes", None)
        if indexes is not None and len(indexes) != len(deepstack):
            raise RuntimeError(
                f"DeepStack 层数量不匹配：config={len(indexes)}, "
                f"实际={len(deepstack)}。"
            )

        # 先验证所有流的 Merger 后结构，避免某条支路失败时悄悄跳过。
        for i, feature in enumerate(deepstack):
            if (
                not isinstance(feature, torch.Tensor)
                or feature.shape != pooled.shape
                or feature.dtype != pooled.dtype
            ):
                raise RuntimeError(
                    f"DeepStack[{i}] 和 pooled 特征无法共享 Adapter："
                    f"pooled={tuple(pooled.shape)}/{pooled.dtype}; "
                    f"deepstack={getattr(feature, 'shape', None)}/"
                    f"{getattr(feature, 'dtype', None)}。"
                )

        def adapt_one(feature, label):
            enhanced = self.res_adapter(
                feature,
                biomed_img_feat=img_f,
                biomed_txt_feat=txt_f,
                grid_thw=grid_thw,
            )
            if enhanced.shape != feature.shape or enhanced.dtype != feature.dtype:
                raise RuntimeError(
                    f"{label} 增强后 shape/dtype 变化："
                    f"before={tuple(feature.shape)}/{feature.dtype}, "
                    f"after={tuple(enhanced.shape)}/{enhanced.dtype}"
                )
            return enhanced

        # 固定顺序：pooled -> DeepStack 0/1/2；评测也按这个顺序记录。
        enhanced_pooled = adapt_one(pooled, "pooler_output")
        enhanced_deepstack = [
            adapt_one(feature, f"deepstack_{i}")
            for i, feature in enumerate(deepstack)
        ]

        if output_format == "model_output":
            outputs.pooler_output = enhanced_pooled
            outputs.deepstack_features = enhanced_deepstack
            return outputs
        if output_format == "tuple":
            return enhanced_pooled, enhanced_deepstack
        return [enhanced_pooled, enhanced_deepstack]

    vision_tower.forward = types.MethodType(patched_vision_forward, vision_tower)
    vision_tower._cm_da_ssm_vision_patched = True
    return vision_tower


# =====================================================================
# 外层 PEFT forward patch：提取并缓存当前 batch 的 BioMedCLIP 条件
# =====================================================================
def install_cm_da_ssm_biomed_forward_patch(
    peft_model: nn.Module,
):
    """
    对齐原 MoRA-Med 的 PeftModel.forward 接口：

        biomed_image_tensors
        biomed_text_tokens

    两个参数不属于 Qwen3-VL 原生 forward，
    因此必须在外层 patch 中 pop 并执行 BioMedCLIP 编码。

    本版在一次 forward 完成后用 finally 清理缓存，避免下一批
    数据误用上一批的条件特征。

    注意生成阶段：
        原项目的评测脚本会预先缓存 BioMedCLIP 特征，
        并 patch generate 所使用的 forward，绕过本函数，
        因此不会每生成一个 token 都重复编码 BioMedCLIP。

        这一机制仍然可用，但旧评估脚本的三专家日志
        需要在后续步骤迁移为四方向统计。
    """
    if getattr(peft_model, "_cm_da_ssm_biomed_patched", False):
        raise RuntimeError(
            "外层 PEFT forward 已经安装 CM-DA-SSM patch。"
        )

    vision_tower = peft_model.base_model.model.model.visual

    if not getattr(
        vision_tower,
        "_cm_da_ssm_vision_patched",
        False,
    ):
        raise RuntimeError(
            "请先挂载 pooled + DeepStack Vision Tower patch，"
            "再安装 BioMedCLIP forward patch。"
        )

    if not hasattr(peft_model, "biomed_extractor"):
        raise RuntimeError(
            "peft_model 未挂载 biomed_extractor。"
        )

    peft_model.original_forward = peft_model.forward

    def patched_model_forward(self, *args, **kwargs):
        # 这里先移除 Qwen 原生 forward 不认识的额外参数。
        biomed_img = kwargs.pop(
            "biomed_image_tensors",
            None,
        )
        biomed_txt = kwargs.pop(
            "biomed_text_tokens",
            None,
        )

        visual = self.base_model.model.model.visual

        # 一次外层 forward 对应一个 batch 的条件。
        visual.current_biomed_img_feat = None
        visual.current_biomed_txt_feat = None

        try:
            if (biomed_img is None) != (biomed_txt is None):
                raise ValueError(
                    "biomed_image_tensors 与 biomed_text_tokens "
                    "必须同时提供，不能只传一个。"
                )

            if biomed_img is not None:
                # 兼容原 MoRA-Med 旧 FrozenBioMedCLIPFeatureExtractor：
                # 即使 model.train() 恢复了外层模块训练模式，
                # 这里也强制使用冻结的 eval 状态。
                self.biomed_extractor.eval()

                # 与 BioMedCLIP 的实际设备和浮点 dtype 对齐。
                # Collator 常返回 FP32 image tensor，而冻结模型可能
                # 已被移动到 BF16；这里显式对齐，避免 Conv/Linear
                # 由于 input / weight dtype 不一致而报错。
                #
                # 文本 token 保持原有整数 dtype，只移动设备。
                biomed_ref_param = next(
                    self.biomed_extractor.parameters(),
                    None,
                )

                if biomed_ref_param is not None:
                    if isinstance(biomed_img, torch.Tensor):
                        image_dtype = (
                            biomed_ref_param.dtype
                            if biomed_ref_param.is_floating_point()
                            else biomed_img.dtype
                        )

                        biomed_img = biomed_img.to(
                            device=biomed_ref_param.device,
                            dtype=image_dtype,
                        )

                    if isinstance(biomed_txt, torch.Tensor):
                        biomed_txt = biomed_txt.to(
                            device=biomed_ref_param.device,
                        )

                with torch.no_grad():
                    img_f, txt_f = self.biomed_extractor(
                        biomed_img,
                        biomed_txt,
                    )

                visual.current_biomed_img_feat = img_f.detach()
                visual.current_biomed_txt_feat = txt_f.detach()

            # 这里保留原始 Qwen/PEFT forward；
            # 不改动图像 merge、DeepStack 注入及 LLM 输入。
            return self.original_forward(
                *args,
                **kwargs,
            )

        finally:
            # 仅清除 BioMedCLIP 条件缓存，不影响增强后的 DeepStack。
            visual.current_biomed_img_feat = None
            visual.current_biomed_txt_feat = None

    peft_model.forward = types.MethodType(
        patched_model_forward,
        peft_model,
    )

    peft_model._cm_da_ssm_biomed_patched = True

    return peft_model


# =====================================================================
# Qwen3-VL + LoRA + CM-DA-SSM Wrapper
# =====================================================================
class Qwen3VLLoraAndCMDASSMWrapper:
    """
    CM-DA-SSM 对应的 Qwen3-VL LoRA Wrapper。

    完整计算：
        Frozen Qwen3-VL Vision Encoder
            -> pooler_output + DeepStack 0/1/2
            -> 同一 CM-DA-SSM Adapter 依次处理四条视觉流
            -> 各流 X' = X + lambda*rho*R
            -> Qwen3-VL 原生 LLM 注入路径 + LoRA

    重要设计：
        - Frozen Qwen backbone + 可训练 LoRA；
        - 四方向 SSM / CM-RADF / Shared Up 可训练；
        - learnable lambda 可训练；
        - BioMedCLIP 完全冻结；
        - pooled 与 DeepStack 共享同一个 Adapter，不复制模型参数；
        - 不包含旧三卷积专家/样本 Soft Gate；
        - 返回的依旧是 PeftModel，可以供 Trainer 使用。

    注意：该结构替换本工程原单路径视觉适配版本；旧版由 Git 历史保存。
    Stage 1/2 必须使用相同的新版结构指纹。
    """

    def __init__(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: List[str],
        gradient_checkpointing: bool,

        # -------------------------------------------------------------
        # CM-DA-SSM Adapter
        # -------------------------------------------------------------
        visual_adapter_hidden_dim: int = 4096,
        visual_adapter_r: int = 16,
        enable_visual_adapter: bool = True,

        # -------------------------------------------------------------
        # Frozen BiomedCLIP
        # -------------------------------------------------------------
        biomed_extractor: Optional[nn.Module] = None,

        # -------------------------------------------------------------
        # Four-Directional SSM
        # -------------------------------------------------------------
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        ssm_backend: str = "mamba",

        # -------------------------------------------------------------
        # Global–Local CM-RADF
        # -------------------------------------------------------------
        biomedclip_image_encoder_dim: int = 512,
        biomedclip_text_encoder_dim: int = 512,
        condition_hidden_dim: int = 128,
        query_dim: int = 64,
        eta: float = 1.0,
        eta_mode: str = "fixed",

        # -------------------------------------------------------------
        # Capped RMS Residual Injection
        # -------------------------------------------------------------
        lambda_init: float = 0.2,
        residual_norm_eps: float = 1e-6,
        residual_norm_ratio_clip: float = 10.0,
        zero_init_up: bool = True,

        # -------------------------------------------------------------
        # 五组消融的执行开关（A0 由 enable_visual_adapter 控制）
        # -------------------------------------------------------------
        fusion_mode: str = "cm_radf",
        lambda_mode: str = "learnable",
        use_rms_norm: bool = True,
    ):
        # LoRA
        self.r = lora_r
        self.alpha = lora_alpha
        self.dropout = lora_dropout
        self.target_modules = lora_target_modules
        self.gradient_checkpointing = gradient_checkpointing

        # Visual Adapter
        self.visual_adapter_hidden_dim = visual_adapter_hidden_dim
        self.visual_adapter_r = visual_adapter_r
        self.enable_visual_adapter = enable_visual_adapter

        # BioMedCLIP
        self.biomed_extractor = biomed_extractor

        # Four-Directional SSM
        self.ssm_d_state = ssm_d_state
        self.ssm_d_conv = ssm_d_conv
        self.ssm_expand = ssm_expand
        self.ssm_backend = ssm_backend

        # CM-RADF
        self.biomedclip_image_encoder_dim = biomedclip_image_encoder_dim
        self.biomedclip_text_encoder_dim = biomedclip_text_encoder_dim
        self.condition_hidden_dim = condition_hidden_dim
        self.query_dim = query_dim
        self.eta = eta
        self.eta_mode = eta_mode
        self.fusion_mode = fusion_mode
        self.lambda_mode = lambda_mode
        self.use_rms_norm = bool(use_rms_norm)

        # RMS + Lambda
        self.lambda_init = lambda_init
        self.residual_norm_eps = residual_norm_eps
        self.residual_norm_ratio_clip = residual_norm_ratio_clip
        self.zero_init_up = zero_init_up

    @staticmethod
    def _patch_forward_to_drop_biomed_kwargs(peft_model: nn.Module):
        """
        A0 / LoRA-only 消融依旧允许 collator 传入 BiomedCLIP 字段，
        但这些字段不应该传给 Qwen 原始 forward。
        """
        peft_model.original_forward = peft_model.forward

        def patched_model_forward(self, *args, **kwargs):
            kwargs.pop(
                "biomed_image_tensors",
                None,
            )

            kwargs.pop(
                "biomed_text_tokens",
                None,
            )

            return self.original_forward(
                *args,
                **kwargs,
            )

        peft_model.forward = types.MethodType(
            patched_model_forward,
            peft_model,
        )

    def wrap(self, model: nn.Module) -> nn.Module:
        """
        保持 MoRA-Med 原来的 LoRA -> Adapter 挂载顺序。

        这里延迟导入 peft，便于在 CPU 测试环境中只检查
        Vision Forward Patch 与残差注入逻辑。

        真正训练仍要求项目原来的 PEFT / Transformers 依赖。
        """
        try:
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError as exc:
            raise ImportError(
                "Qwen3VL LoRA Wrapper 需要 peft；"
                "请在原 MoRA-Med 的训练环境安装对应版本。"
            ) from exc

        # ============================================================
        # 1. 准备 QLoRA / LoRA
        # ============================================================
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.gradient_checkpointing,
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            },
        )

        if hasattr(model, "config"):
            model.config.use_cache = False

        peft_model = get_peft_model(
            model,
            LoraConfig(
                r=self.r,
                lora_alpha=self.alpha,
                target_modules=self.target_modules,
                lora_dropout=self.dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()

        # ============================================================
        # 2. A0 / LoRA-only 直接返回
        # ============================================================
        if not self.enable_visual_adapter:
            self._patch_forward_to_drop_biomed_kwargs(
                peft_model
            )

            ddp_print(
                "\n✅ [CM-DA-SSM A0] LoRA-only，"
                "未挂载视觉残差适配器。"
            )

            if hasattr(peft_model, "print_trainable_parameters"):
                peft_model.print_trainable_parameters()

            return peft_model

        if self.fusion_mode == "cm_radf" and self.biomed_extractor is None:
            raise ValueError(
                "A2/A3/A3-NoRMS 启用 CM-RADF 时必须提供 "
                "冻结的 BiomedCLIP 编码器。"
            )

        # A1 Uniform：不使用 Query/Key/图文先验，禁止无意义加载
        # BiomedCLIP 并参与 Forward。即使调用者传了 extractor，
        # 这里也不将其注册到 PEFT model。

        # ============================================================
        # 3. 定位 Qwen3-VL Vision Tower
        # ============================================================
        #
        # 这里保留原 MoRA-Med 精确路径，
        # 不通过不确定的字符串查找视觉层。
        try:
            vision_tower = peft_model.base_model.model.model.visual
        except AttributeError as exc:
            raise RuntimeError(
                "无法定位 Qwen3-VL Vision Tower："
                "预期路径为 peft_model.base_model.model.model.visual。"
                "请核对 Qwen3-VL / PEFT 版本。"
            ) from exc

        ref_param = next(
            vision_tower.parameters(),
            None,
        )

        if ref_param is None:
            raise RuntimeError(
                "Vision Tower 不包含可用于确定 device/dtype 的参数。"
            )

        adapter_device = ref_param.device

        adapter_dtype = (
            ref_param.dtype
            if ref_param.is_floating_point()
            else torch.bfloat16
        )

        ddp_print(
            "\n✨ [CM-DA-SSM] 正在挂载 "
            "Four-Directional Selective SSM + "
            "Global–Local CM-RADF + Capped RMS Adapter..."
        )

        ddp_print(
            f"    hidden_dim={self.visual_adapter_hidden_dim}, "
            f"r={self.visual_adapter_r}, "
            f"ssm_backend={self.ssm_backend}, "
            f"eta_mode={self.eta_mode}, "
            f"eta={self.eta}, "
            f"lambda_init={self.lambda_init}, "
            f"rho_max={self.residual_norm_ratio_clip}, "
            "adapt_deepstack=True (shared adapter)"
        )

        # ============================================================
        # 4. 实例化新的完整视觉残差适配器
        # ============================================================
        visual_adapter = Qwen3VLCMDASSMVisualAdapter(
            hidden_dim=self.visual_adapter_hidden_dim,
            r=self.visual_adapter_r,

            ssm_d_state=self.ssm_d_state,
            ssm_d_conv=self.ssm_d_conv,
            ssm_expand=self.ssm_expand,
            ssm_backend=self.ssm_backend,

            biomedclip_image_encoder_dim=(
                self.biomedclip_image_encoder_dim
            ),
            biomedclip_text_encoder_dim=(
                self.biomedclip_text_encoder_dim
            ),
            condition_hidden_dim=self.condition_hidden_dim,
            query_dim=self.query_dim,
            eta=self.eta,
            eta_mode=self.eta_mode,

            lambda_init=self.lambda_init,
            residual_norm_eps=self.residual_norm_eps,
            residual_norm_ratio_clip=self.residual_norm_ratio_clip,
            zero_init_up=self.zero_init_up,

            # 传入真实消融执行规则，不靠实验 ID 命名模拟。
            fusion_mode=self.fusion_mode,
            lambda_mode=self.lambda_mode,
            use_rms_norm=self.use_rms_norm,
        )

        # 只将需要参与混合精度计算的 feature_adapter 转为
        # Qwen Vision Tower dtype，保留 lambda_a 为 FP32。
        visual_adapter = visual_adapter.to(
            device=adapter_device,
        )
        visual_adapter.feature_adapter.to(
            dtype=adapter_dtype,
        )

        # ============================================================
        # 5. 第一重 Patch：Vision Tower pooled + DeepStack
        # ============================================================
        install_cm_da_ssm_vision_patch(
            vision_tower=vision_tower,
            visual_adapter=visual_adapter,
        )

        # ============================================================
        # 6. A1 或 A2/A3 的外层 Forward 选择
        # ============================================================
        if self.fusion_mode == "uniform":
            # A1 不需要 BioMedCLIP。因此不注册冻结编码器，
            # 并且移除 Collator 可能携带的额外图文参数。
            #
            # Vision patch 同时更新 pooled 与 DeepStack。
            self._patch_forward_to_drop_biomed_kwargs(
                peft_model
            )
        else:
            # A2 / A3 / A3-NoRMS 才加载图文先验并缓存当前 batch。
            peft_model.biomed_extractor = (
                self.biomed_extractor.to(
                    device=adapter_device,
                    dtype=adapter_dtype,
                )
            )

            peft_model.biomed_extractor.requires_grad_(False)
            peft_model.biomed_extractor.eval()

            install_cm_da_ssm_biomed_forward_patch(
                peft_model=peft_model,
            )

        ddp_print(
            "✅ [CM-DA-SSM] pooled + DeepStack 共享视觉适配器挂载成功。"
        )

        ddp_print(
            "    Enhanced: pooler_output (shared res_adapter)"
        )

        ddp_print(
            "    Enhanced: deepstack_features (same res_adapter, 3 passes)"
        )

        if hasattr(peft_model, "print_trainable_parameters"):
            peft_model.print_trainable_parameters()

        return peft_model