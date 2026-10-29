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
# 视觉塔 forward patch：只注入 pooled visual tokens
# =====================================================================
def install_cm_da_ssm_pooled_only_vision_patch(
    vision_tower: nn.Module,
    visual_adapter: Qwen3VLCMDASSMVisualAdapter,
):
    """
    在已有 Qwen3-VL Vision Tower 上挂载完整的 CM-DA-SSM。

    这一步严格遵循原 MoRA-Med 的挂载位置：
        vision_tower.res_adapter = ...

    同时修改 vision_tower.forward：
        1. 原生 Qwen Vision Encoder 先正常执行；
        2. 取得 merger 后的 pooled tokens（支持对象与二元组）；
        3. 只把 pooled tokens 传入 CM-DA-SSM；
        4. 将输出 X' 回填到原生输出结构的 pooled 位置；
        5. 按原生格式返回（ModelOutput / tuple / list）。

    本函数不会对原生 DeepStack：
        - 调用任何新模块；
        - 进行 DeepStack 列表重建；
        - 执行 clone / detach；
        - 修改其对象引用。

    因此 DeepStack 张量及容器的身份、数值、梯度路径
    均由原生 Qwen3-VL 决定，不会被本 patch 改变。

    返回：
        被原地打补丁的同一个 vision_tower 对象。

    注意：
        不支持重复打补丁，也不覆盖已经存在的旧版 res_adapter。
        在同一模型实例上切换旧 / 新架构应重新构建 Qwen3-VL。
    """
    if getattr(vision_tower, "_cm_da_ssm_pooled_only_patched", False):
        raise RuntimeError(
            "当前 Vision Tower 已经安装 CM-DA-SSM patch，"
            "请勿重复挂载。"
        )

    if hasattr(vision_tower, "res_adapter"):
        raise RuntimeError(
            "当前 Vision Tower 已存在 res_adapter，"
            "请使用全新的 Qwen3-VL 模型实例安装 CM-DA-SSM，"
            "避免与旧 MoRA-Med forward patch 冲突。"
        )

    if not isinstance(visual_adapter, nn.Module):
        raise TypeError(
            "visual_adapter 必须是 nn.Module。"
        )

    # 挂到视觉塔下，确保 adapter 参数被 PEFT / PyTorch 追踪，
    # 并使原项目 visual.res_adapter 的保存路径仍然有效。
    vision_tower.res_adapter = visual_adapter

    # 保存未注入前的真正 Qwen3-VL 原生 vision forward。
    vision_tower.original_forward = vision_tower.forward

    # 默认不缓存任何前一个 batch 的条件特征。
    vision_tower.current_biomed_img_feat = None
    vision_tower.current_biomed_txt_feat = None

    def patched_vision_forward(self, *args, **kwargs):
        """
        每次 Qwen 视觉塔前向执行：

        native vision -> pooled-only CM-DA-SSM -> native DeepStack

        DeepStack 不传给 res_adapter，也不重新赋值。
        """
        # ============================================================
        # 1. 保持 Qwen3-VL Vision Encoder 原生计算
        # ============================================================
        outputs = self.original_forward(
            *args,
            **kwargs,
        )

        # ============================================================
        # 2. 查找当前 batch 的 BioMedCLIP 条件特征
        # ============================================================
        img_f = getattr(
            self,
            "current_biomed_img_feat",
            None,
        )

        txt_f = getattr(
            self,
            "current_biomed_txt_feat",
            None,
        )

        # A1 的 Uniform Fusion 根本不使用 BiomedCLIP Query，
        # 所以 A1 允许二者均为 None，A2/A3 则必须提供完整先验。
        if (
            self.res_adapter.fusion_mode == "cm_radf"
            and (img_f is None or txt_f is None)
        ):
            raise ValueError(
                "CM-DA-SSM 视觉塔已启用，但当前视觉前向没有 "
                "BioMedCLIP 图像/文本特征。"
                "训练时请传 biomed_image_tensors 和 biomed_text_tokens；"
                "generate() 时请先计算特征并注入 Vision Tower 缓存。"
            )

        # ============================================================
        # 3. 查找 Qwen 原生 grid_thw
        # ============================================================
        #
        # 对齐原 MoRA-Med Wrapper：
        #   grid_thw 可在 keyword 参数或第二个位置参数中。
        grid_thw = kwargs.get(
            "grid_thw",
            None,
        )

        if grid_thw is None and len(args) > 1:
            grid_thw = args[1]

        if grid_thw is None:
            raise ValueError(
                "CM-DA-SSM 视觉前向没有拿到 grid_thw。"
            )

        # ============================================================
        # 4. 只处理原生 pooled visual tokens
        # ============================================================
        #
        # merged/pooler tokens:
        #   [N_total, D] 或 [1, N_total, D]
        #
        # 原生 DeepStack features：
        #   不参与 SSM；
        #   不参与 CM-RADF；
        #   不参与 RMS 校准；
        #   不参与残差注入；
        #   不修改对象引用。
        # Qwen3-VL 不同 Transformers 版本的 Vision Tower
        # 返回接口并不相同：
        #
        #   当前训练环境：
        #       (merged_hidden_states, deepstack_feature_lists)
        #
        #   另一些版本：
        #       BaseModelOutputWithDeepstackFeatures(
        #           pooler_output=merged_hidden_states,
        #           deepstack_features=deepstack_feature_lists,
        #       )
        #
        # 无论哪种接口，只允许增强 merger 后的 pooled tokens；
        # DeepStack 列表及其中各张量必须保持原生对象引用。
        if hasattr(outputs, "pooler_output"):
            output_format = "model_output"
            pooled_tokens = outputs.pooler_output
        elif isinstance(outputs, (tuple, list)):
            if (
                len(outputs) != 2
                or not isinstance(outputs[0], torch.Tensor)
                or not isinstance(outputs[1], (tuple, list))
            ):
                raise RuntimeError(
                    "不支持的 Qwen3-VL Vision tuple/list 输出："
                    f"type={type(outputs).__name__}, "
                    f"len={len(outputs)}。预期为 "
                    "(pooled_tokens, deepstack_features)。"
                )
            output_format = (
                "tuple" if isinstance(outputs, tuple) else "list"
            )
            pooled_tokens = outputs[0]
        else:
            raise RuntimeError(
                "当前 Qwen3-VL Vision 输出不包含 pooler_output，"
                "也不是 (pooled_tokens, deepstack_features) 二元组；"
                f"实际类型：{type(outputs).__name__}。"
            )

        if (
            not isinstance(pooled_tokens, torch.Tensor)
            or pooled_tokens.ndim not in (2, 3)
        ):
            raise RuntimeError(
                "Qwen Vision pooled tokens 类型或维度不正确："
                f"type={type(pooled_tokens).__name__}, "
                f"shape={getattr(pooled_tokens, 'shape', None)}。"
            )

        # 这是整个 wrapper 里唯一的一次 res_adapter 前向。
        enhanced_pooled_tokens = self.res_adapter(
            pooled_tokens,
            biomed_img_feat=img_f,
            biomed_txt_feat=txt_f,
            grid_thw=grid_thw,
        )

        if (
            enhanced_pooled_tokens.shape
            != pooled_tokens.shape
        ):
            raise RuntimeError(
                "CM-DA-SSM 改变了 pooled visual token 的形状："
                f"before={tuple(pooled_tokens.shape)}, "
                f"after={tuple(enhanced_pooled_tokens.shape)}"
            )

        if (
            enhanced_pooled_tokens.dtype
            != pooled_tokens.dtype
        ):
            raise RuntimeError(
                "CM-DA-SSM 改变了 pooled visual token dtype："
                f"before={pooled_tokens.dtype}, "
                f"after={enhanced_pooled_tokens.dtype}"
            )

        # ============================================================
        # 5. 唯一需要更新的视觉输出字段
        # ============================================================
        if output_format == "model_output":
            outputs.pooler_output = enhanced_pooled_tokens
            return outputs

        # 当前 transformers 版本直接解包视觉塔的二元组。
        # 因此必须保留原生的二元组/列表格式：
        #
        #   image_embeds, deepstack_image_embeds = self.visual(...)
        #
        # 只替换第 0 项（pooled tokens），第 1 项 DeepStack
        # 不进行 clone、detach、转换或重建，保证其对象身份不变。
        if output_format == "tuple":
            return (enhanced_pooled_tokens, outputs[1])

        return [enhanced_pooled_tokens, outputs[1]]

    vision_tower.forward = types.MethodType(
        patched_vision_forward,
        vision_tower,
    )

    vision_tower._cm_da_ssm_pooled_only_patched = True

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
        "_cm_da_ssm_pooled_only_patched",
        False,
    ):
        raise RuntimeError(
            "请先挂载 pooled-only Vision Tower patch，"
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
            # 仅清除条件缓存，不触碰任何 DeepStack tensor。
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
            -> pooled tokens X + native DeepStack
            -> CM-DA-SSM(X, BioMedCLIP(I,Q))
            -> X' = X + lambda*rho*R
            -> Qwen3-VL LLM + LoRA

    重要设计：
        - Frozen Qwen backbone + 可训练 LoRA；
        - 四方向 SSM / CM-RADF / Shared Up 可训练；
        - learnable lambda 可训练；
        - BioMedCLIP 完全冻结；
        - DeepStack 完全保留原样；
        - 不包含旧三卷积专家/样本 Soft Gate；
        - 返回的依旧是 PeftModel，可以供 Trainer 使用。

    注意：
        此类与原 MoRA-Med 的 Qwen3VLLoraAndVisualAdapterWrapper
        并行存在，不覆盖旧代码，保证旧基线可复现。

        旧 Trainer 的 scale_mode/gate_mode 等参数不能原样传入，
        必须在后续训练迁移时更新配置与优化器分组。
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
            "adapt_deepstack=False"
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
        # 5. 第一重 Patch：Vision Tower pooled-only
        # ============================================================
        install_cm_da_ssm_pooled_only_vision_patch(
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
            # Vision patch 仍只更新 pooled tokens，DeepStack 不变。
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
            "✅ [CM-DA-SSM] pooled-only 视觉残差适配器挂载成功。"
        )

        ddp_print(
            "    Enhanced: pooler_output"
        )

        ddp_print(
            "    Unmodified: deepstack_features"
        )

        if hasattr(peft_model, "print_trainable_parameters"):
            peft_model.print_trainable_parameters()

        return peft_model