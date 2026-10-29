"""
CM-DA-SSM 五组消融实验共享配置。

以论文「八、最终消融组合与执行安排」为唯一消融定义：
    A0        : Qwen3-VL + LoRA，无新增 SSM
    A1        : Four-Directional SSM + Uniform Fusion + Capped RMS + 固定 λ=0.4
    A2        : Four-Directional SSM + CM-RADF       + Capped RMS + 固定 λ=0.4
    A3        : Four-Directional SSM + CM-RADF       + Capped RMS + 可学习 λ(0)=0.4
    A3-NoRMS  : Four-Directional SSM + CM-RADF       + ρ≡1       + 可学习 λ(0)=0.4

设计原则：
    1. 每组 Stage 1 独立从同一个 Qwen 基座初始化并训练；
    2. 每组 Stage 2 只继承该组 Stage 1 的 LoRA + Adapter；
    3. A1 不加载 BioMedCLIP，因不需要 Query；
    4. A3 与 A3-NoRMS 的初始结构、随机种子、LR、训练步数可配对比较；
    5. 输出路径必须包含消融名、结构指纹、随机种子及实验标签；
    6. 配置负责验证与生成路径，不执行权重加载或训练。

旧 MoRA-Med 的 A0～A6 定义不再使用；本文件只保留新五组。
"""

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any


ABLATION_IDS = (
    "A0",
    "A1",
    "A2",
    "A3",
    "A3-NoRMS",
)


# 各项是算法开关，而不是仅用于命名的字符串。
# 字典中的字段将在 __post_init__ 里设置为不可由构造参数覆盖的有效值。
ABLATION_RULES: dict[str, dict[str, Any]] = {
    "A0": {
        "enable_visual_adapter": False,
        "fusion_mode": "none",
        "lambda_mode": "none",
        "use_rms_norm": False,
        "requires_biomedclip": False,
    },
    "A1": {
        "enable_visual_adapter": True,
        "fusion_mode": "uniform",
        "lambda_mode": "fixed",
        "use_rms_norm": True,
        "requires_biomedclip": False,
    },
    "A2": {
        "enable_visual_adapter": True,
        "fusion_mode": "cm_radf",
        "lambda_mode": "fixed",
        "use_rms_norm": True,
        "requires_biomedclip": True,
    },
    "A3": {
        "enable_visual_adapter": True,
        "fusion_mode": "cm_radf",
        "lambda_mode": "learnable",
        "use_rms_norm": True,
        "requires_biomedclip": True,
    },
    "A3-NoRMS": {
        "enable_visual_adapter": True,
        "fusion_mode": "cm_radf",
        "lambda_mode": "learnable",
        "use_rms_norm": False,
        "requires_biomedclip": True,
    },
}


def validate_run_tag(value: str, field_name: str) -> None:
    """实验标签仅允许安全字符，杜绝路径分隔符/意外覆盖。"""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value) is None
    ):
        raise ValueError(
            f"{field_name} 必须为 1~48 位字母、数字、_ 或 -，当前为 {value!r}"
        )


@dataclass
class CMDASSMCommonConfig:
    """
    Stage 1/2 共同遵守的模型结构、消融逻辑及输出命名。

    Stage 2 可以覆盖训练学习率/数据集设置，但不应修改
    用于继承 Stage 1 权重的架构字段。
    """

    # =========================================================
    # 1. 基本信息和模型根目录
    # =========================================================
    ablation_id: str = "A3"
    seed: int = 2048
    print_rank: int = 0
    run_tag: str = "main"

    # 与旧 MoRA-Med 权重目录隔离；所有五组共用一个项目根路径。
    output_root: str = "/home/yuqing/Models/CM_DA_SSM"

    # =========================================================
    # 2. Qwen3-VL / QLoRA
    # =========================================================
    model_name_or_path: str = "/home/yuqing/Models/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = True

    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # =========================================================
    # 3. Frozen BioMedCLIP
    # =========================================================
    biomedclip_path: str = (
        "/home/yuqing/Models/"
        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    biomedclip_image_encoder_dim: int = 512
    biomedclip_text_encoder_dim: int = 512

    # =========================================================
    # 4. Four-Directional SSM
    # =========================================================
    visual_adapter_hidden_dim: int = 4096
    visual_adapter_r: int = 16
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    ssm_backend: str = "mamba"

    # =========================================================
    # 5. CM-RADF Query / Key
    # =========================================================
    condition_hidden_dim: int = 128
    query_dim: int = 64
    # η 仅作用于 CM-RADF 组（A2、A3、A3-NoRMS），其余组不使用。
    eta: float = 0.5
    eta_mode: str = "fixed"

    # =========================================================
    # 6. 残差注入
    # =========================================================
    lambda_init: float = 0.4
    residual_norm_eps: float = 1e-6
    residual_norm_ratio_clip: float = 10.0
    zero_init_up: bool = True

    # =========================================================
    # 7. 五组消融开关：只通过 ablation_id 推导！
    # =========================================================
    #
    # init=False 很重要，防止出现名称 A3-NoRMS 却仍计算 RMS、
    # 或名称 A1 却实际运行 CM-RADF 的不一致实验。
    enable_visual_adapter: bool = field(init=False)
    fusion_mode: str = field(init=False)
    lambda_mode: str = field(init=False)
    use_rms_norm: bool = field(init=False)
    requires_biomedclip: bool = field(init=False)

    # 这两个约束是算法固定要求。
    adapt_pooled_tokens: bool = field(default=True, init=False)
    adapt_deepstack: bool = field(default=False, init=False)

    # =========================================================
    # 8. 训练策略：五组默认同一任务预算与数据
    # =========================================================
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 1.0
    max_steps: int = -1
    dataloader_num_workers: int = 12

    # 五组都使用同一套新 AdamW 规则：A0 仅创建 LoRA 组。
    use_discriminative_lr: bool = True
    lr_recipe_name: str = "CMDA-DLR"
    learning_rate: float = 2e-5
    lora_learning_rate: float = 2e-5
    ssm_learning_rate: float = 1e-4
    visual_projection_learning_rate: float = 1e-4
    cm_radf_learning_rate: float = 5e-5
    lambda_learning_rate: float = 5e-5
    visual_norm_learning_rate: float = 2e-5
    weight_decay: float = 0.01

    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2

    def __post_init__(self) -> None:
        """固定真实消融模式，然后执行严格配置校验。"""
        if self.ablation_id not in ABLATION_RULES:
            raise ValueError(
                f"不支持的 ablation_id={self.ablation_id!r}，"
                f"只允许：{ABLATION_IDS}"
            )

        for key, value in ABLATION_RULES[self.ablation_id].items():
            setattr(self, key, value)

        self.validate_common()

    def validate_common(self) -> None:
        """不访问数据集/预训练文件，仅验证数值和结构。"""
        for name in (
            "seed",
            "lora_r",
            "lora_alpha",
            "visual_adapter_hidden_dim",
            "visual_adapter_r",
            "ssm_d_state",
            "ssm_d_conv",
            "ssm_expand",
            "condition_hidden_dim",
            "query_dim",
            "biomedclip_image_encoder_dim",
            "biomedclip_text_encoder_dim",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    f"{name} 必须是 >= {minimum} 的整数，实际 {value!r}"
                )

        if self.visual_adapter_hidden_dim % self.visual_adapter_r != 0:
            raise ValueError(
                "hidden_dim 必须可被 bottleneck ratio 整除。"
            )

        if self.ssm_backend not in {"mamba", "torch_reference"}:
            raise ValueError(
                "ssm_backend 只能为 mamba 或 torch_reference。"
            )

        if (
            self.biomedclip_image_encoder_dim != 512
            or self.biomedclip_text_encoder_dim != 512
        ):
            raise ValueError(
                "当前 2049 维先验要求 Image/Text 特征各 512 维。"
            )

        if self.eta_mode not in {"fixed", "learnable"}:
            raise ValueError("eta_mode 必须为 fixed/learnable。")
        if not math.isfinite(float(self.eta)) or self.eta < 0:
            raise ValueError("eta 必须是非负有限数。")
        if self.eta_mode == "learnable" and self.eta <= 0:
            raise ValueError("learnable eta 必须 > 0。")

        if not math.isfinite(float(self.lambda_init)):
            raise ValueError("lambda_init 必须为有限数。")

        # 新一轮五组消融统一使用 lambda_init=0.4：
        # A1/A2 固定 λ=0.4，A3/A3-NoRMS 使用可学习 λ(0)=0.4。
        # 保持 A1/A2/A3 的 λ 起点一致，避免结构消融混入初值差异。
        if not math.isclose(float(self.lambda_init), 0.4, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                "本轮五组消融统一使用 lambda_init=0.4；"
                "比较其他初值请另建独立调参实验。"
            )

        if self.zero_init_up is not True:
            raise ValueError(
                "五组主实验要求 Shared Up 零初始化，"
                "才能确保各视觉适配器初始为恒等映射。"
            )

        for name in (
            "residual_norm_eps",
            "residual_norm_ratio_clip",
            "learning_rate",
            "lora_learning_rate",
            "ssm_learning_rate",
            "visual_projection_learning_rate",
            "cm_radf_learning_rate",
            "lambda_learning_rate",
            "visual_norm_learning_rate",
            "num_train_epochs",
        ):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0:
                raise ValueError(
                    f"{name} 必须为正有限数，实际 {val!r}。"
                )

        if (
            not math.isfinite(float(self.weight_decay))
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay 必须为非负有限数。")

        if self.max_steps != -1 and self.max_steps <= 0:
            raise ValueError("max_steps 只能为 -1 或正整数。")

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps 不能为负数。")

        if not self.lora_target_modules:
            raise ValueError("lora_target_modules 不能为空。")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout 必须位于 [0,1)。")

        if not self.use_discriminative_lr:
            raise ValueError(
                "本五组实验采用统一的分组规则。"
                "如需等学习率实验请另建独立对照，不要混入这五组。"
            )

        validate_run_tag(self.run_tag, "run_tag")

    @property
    def bottleneck_dim(self) -> int:
        return self.visual_adapter_hidden_dim // self.visual_adapter_r

    @property
    def ablation_rule(self) -> dict[str, Any]:
        """返回当前实验实际运行的开关副本，供日志输出。"""
        return dict(ABLATION_RULES[self.ablation_id])

    def architecture_parameters(self) -> dict[str, Any]:
        """
        Stage 1/2 必须一致的 checkpoint 结构及模式定义。

        A3-NoRMS 与 A3 虽然参数 shape 相同，但实验路径必须不同，
        因此 ablation_id 与 use_rms_norm 均属于架构指纹。
        """
        return {
            "version": "CM-DA-SSM-5AB-V1",
            "ablation_id": self.ablation_id,
            "model_name_or_path": self.model_name_or_path,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
            "enable_visual_adapter": self.enable_visual_adapter,
            "fusion_mode": self.fusion_mode,
            "lambda_mode": self.lambda_mode,
            "lambda_init": self.lambda_init,
            "use_rms_norm": self.use_rms_norm,
            "residual_norm_eps": self.residual_norm_eps,
            "residual_norm_ratio_clip": self.residual_norm_ratio_clip,
            "zero_init_up": self.zero_init_up,
            "hidden_dim": self.visual_adapter_hidden_dim,
            "r": self.visual_adapter_r,
            "ssm_d_state": self.ssm_d_state,
            "ssm_d_conv": self.ssm_d_conv,
            "ssm_expand": self.ssm_expand,
            "ssm_backend": self.ssm_backend,
            "biomed_image_dim": self.biomedclip_image_encoder_dim,
            "biomed_text_dim": self.biomedclip_text_encoder_dim,
            "condition_hidden_dim": self.condition_hidden_dim,
            "query_dim": self.query_dim,
            "eta": self.eta,
            "eta_mode": self.eta_mode,
            "adapt_pooled_tokens": self.adapt_pooled_tokens,
            "adapt_deepstack": self.adapt_deepstack,
        }

    @property
    def architecture_fingerprint(self) -> str:
        """SHA256 的前 12 位，仅用于防止实验名称/参数串线。"""
        text = json.dumps(
            self.architecture_parameters(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _path_number(value: float | int) -> str:
        """
        将数值转换为适合目录名的紧凑形式。

        例如：0.2 -> 0p2；0.05 -> 0p05；1e-6 -> 1em06。
        与目录名中的 '-' 分隔符区分开，避免解析歧义。
        """
        return (
            f"{value:g}"
            .replace(".", "p")
            .replace("-", "m")
            .replace("+", "")
        )

    @property
    def architecture_tag(self) -> str:
        """
        MoRA-Med 风格的显式架构目录名，由真实配置动态生成。

        例：A3_4DSSM-mamba-H4096-R16-S16-C4-E2_..._Sig6c710d72

        保留末尾 8 位结构指纹（不再作为主要目录信息），防止：
          - LoRA target modules 等未全部展示的架构参数变化；
          - 同名配置误覆盖、跨消融混用 checkpoint。

        注意：Stage 1 与所有 Stage 2 都使用这个属性，所以
        Stage 2 的 stage1_weights_dir 会自动对应新目录。
        """
        num = self._path_number

        # A0 没有视觉残差适配器，路径不能伪称包含 SSM、CM-RADF、
        # λ 或 RMS；只列出 A0 实际使用的模块。
        fragments = [self.ablation_id]

        if self.enable_visual_adapter:
            fragments.append(
                f"4DSSM-{self.ssm_backend}"
                f"-H{self.visual_adapter_hidden_dim}"
                f"-R{self.visual_adapter_r}"
                f"-S{self.ssm_d_state}"
                f"-C{self.ssm_d_conv}"
                f"-E{self.ssm_expand}"
            )

            if self.fusion_mode == "uniform":
                fragments.append("Fusion-Uniform1over4")
            elif self.fusion_mode == "cm_radf":
                fragments.append(
                    f"Fusion-CMRADF"
                    f"-Cond{self.condition_hidden_dim}"
                    f"-Q{self.query_dim}"
                    f"-Eta{self.eta_mode.capitalize()}{num(self.eta)}"
                )
            else:
                raise ValueError(
                    f"未知融合模式：{self.fusion_mode!r}"
                )

            if self.lambda_mode == "fixed":
                fragments.append(
                    f"LambdaFixed{num(self.lambda_init)}"
                )
            elif self.lambda_mode == "learnable":
                fragments.append(
                    f"LambdaLearnInit{num(self.lambda_init)}"
                )
            else:
                raise ValueError(
                    f"未知 lambda 模式：{self.lambda_mode!r}"
                )

            fragments.append(
                (
                    f"CappedRMS-Clip{num(self.residual_norm_ratio_clip)}"
                    f"-Eps{num(self.residual_norm_eps)}"
                )
                if self.use_rms_norm
                else "NoRMS"
            )
            fragments.append(
                f"UpZero{int(self.zero_init_up)}"
                f"-PooledOnly{int(self.adapt_pooled_tokens)}"
            )

        fragments.append(
            f"LoRA-R{self.lora_r}"
            f"-A{self.lora_alpha}"
            f"-Drop{num(self.lora_dropout)}"
        )

        # BioMedCLIP 是跨模态先验，不是所有消融都使用。
        fragments.append(
            "BioMedCLIP1" if self.requires_biomedclip
            else "BioMedCLIP0"
        )

        # QLoRA 精度/量化策略放在路径中，避免混淆量化实验。
        if self.load_in_4bit:
            fragments.append(
                f"QLoRA-{self.bnb_4bit_quant_type}"
                f"-{self.bnb_4bit_compute_dtype}"
            )
        else:
            fragments.append("Non4bit")

        # 保留短指纹只是保险：可读参数是目录的主体。
        fragments.append(
            f"Sig{self.architecture_fingerprint[:8]}"
        )
        label = "_".join(fragments)

        # Linux 文件系统对单一文件名通常限制 255 字节。
        if len(label.encode("utf-8")) > 255:
            raise ValueError(
                "architecture_tag 超过 255 字节；"
                "请将更多细节写入 architecture.json，"
                "或缩短目录标签。"
            )

        return label

    @staticmethod
    def _learning_rate_number(value: float) -> str:
        """
        将实际学习率转为可读且适合文件名的科学计数法。

        示例：
            2e-5 -> 2em5
            1e-4 -> 1em4
            5e-5 -> 5em5
            1.25e-4 -> 1p25em4

        用 'em' 表示负指数，避免带 '-' 的路径标签产生歧义。
        """
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"学习率必须是正的有限数：{value!r}"
            )

        mantissa, exponent_text = f"{value:.12e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
        exponent = int(exponent_text)
        if exponent < 0:
            return f"{mantissa}em{abs(exponent)}"
        return f"{mantissa}e{exponent}"

    @property
    def lr_tag(self) -> str:
        """
        使用当前对象中的真实分组学习率动态生成目录名。

        注意：每一个 lr_tag 都作为独立目录层级，不与
        architecture_tag 拼成同一个文件夹名称，避免超过
        Linux 常见的 255 字节单个文件名限制。

        A0 只展示实际存在的 LoRA 训练组；
        A1 没有 CM-RADF Router/Lambda 参数组；
        A2 固定 λ 不写 LambdaLR；
        A3/A3-NoRMS 才有可学习 λ 学习率；
        A3-NoRMS 依然有 LN 参数，仍须写 VisualNormLR。
        """
        n = self._learning_rate_number
        parts = [
            self.lr_recipe_name,
            f"LoRALR{n(self.lora_learning_rate)}",
        ]

        if self.enable_visual_adapter:
            parts.extend([
                f"SSMLR{n(self.ssm_learning_rate)}",
                f"VisualProjLR{n(self.visual_projection_learning_rate)}",
            ])

            # A1 的 Uniform Fusion 没有 CM-RADF Router 参数组；
            # A2/A3/A3-NoRMS 才有 cm_radf 学习率。
            if self.fusion_mode == "cm_radf":
                parts.append(f"CMRADFLR{n(self.cm_radf_learning_rate)}")

            # A1/A2 的 lambda 为固定 buffer，没有学习率；
            # A3/A3-NoRMS 才有 lambda_a 可训练参数。
            if self.lambda_mode == "learnable":
                parts.append(f"LambdaLR{n(self.lambda_learning_rate)}")

            # 即使是 A3-NoRMS，Adapter 仍有可训练 LayerNorm
            # (ln_in、ln_dir、ln_fuse)，所以始终有 visual_norm 组。
            parts.append(f"VisualNormLR{n(self.visual_norm_learning_rate)}")

        label = "_".join(parts)
        if len(label.encode("utf-8")) > 255:
            raise ValueError(
                "学习率目录标签超过 255 字节，请缩短 LR 字段名。"
            )
        return label

    @staticmethod
    def stage1_run_dir(
        output_root: str,
        cache_prefix: str,
        architecture_tag: str,
        seed: int,
        lr_tag: str,
        run_tag: str,
    ) -> str:
        """所有 Stage 1 权重目录只通过这一函数生成。"""
        return os.path.join(
            output_root,
            "Stage1_MIMIC_CXR",
            cache_prefix,
            architecture_tag,
            lr_tag,
            f"seed_{seed}",
            f"run_{run_tag}",
        )

    def optimizer_kwargs(self, training_args=None) -> dict[str, Any]:
        """
        与新版 build_discriminative_adamw() 一致的参数字典。

        使用示例：
            optimizer = build_discriminative_adamw(
                model,
                **cfg.optimizer_kwargs(training_args),
            )

        A0 同样可调用；优化器只会生成 LoRA 组。
        """
        return {
            "ablation_id": self.ablation_id,
            "lora_learning_rate": self.lora_learning_rate,
            "ssm_learning_rate": self.ssm_learning_rate,
            "visual_projection_learning_rate": (
                self.visual_projection_learning_rate
            ),
            "cm_radf_learning_rate": self.cm_radf_learning_rate,
            "lambda_learning_rate": self.lambda_learning_rate,
            "visual_norm_learning_rate": self.visual_norm_learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": getattr(training_args, "adam_beta1", 0.9),
            "adam_beta2": getattr(training_args, "adam_beta2", 0.999),
            "adam_epsilon": getattr(training_args, "adam_epsilon", 1e-8),
        }

    def wrapper_kwargs(self, *, biomed_extractor=None) -> dict[str, Any]:
        """
        与现有 Qwen3VLLoraAndCMDASSMWrapper.__init__ 完全匹配。

        A0/A1 不需要 BioMedCLIP；
        A2/A3/A3-NoRMS 才需要传入 Frozen BiomedCLIP。
        """
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
            "gradient_checkpointing": self.gradient_checkpointing,
            "visual_adapter_hidden_dim": self.visual_adapter_hidden_dim,
            "visual_adapter_r": self.visual_adapter_r,
            "enable_visual_adapter": self.enable_visual_adapter,
            "biomed_extractor": (
                biomed_extractor if self.requires_biomedclip else None
            ),
            "ssm_d_state": self.ssm_d_state,
            "ssm_d_conv": self.ssm_d_conv,
            "ssm_expand": self.ssm_expand,
            "ssm_backend": self.ssm_backend,
            "biomedclip_image_encoder_dim": self.biomedclip_image_encoder_dim,
            "biomedclip_text_encoder_dim": self.biomedclip_text_encoder_dim,
            "condition_hidden_dim": self.condition_hidden_dim,
            "query_dim": self.query_dim,
            "eta": self.eta,
            "eta_mode": self.eta_mode,
            "lambda_init": self.lambda_init,
            "residual_norm_eps": self.residual_norm_eps,
            "residual_norm_ratio_clip": self.residual_norm_ratio_clip,
            "zero_init_up": self.zero_init_up,
            "fusion_mode": (
                self.fusion_mode if self.enable_visual_adapter else "cm_radf"
            ),
            "lambda_mode": (
                self.lambda_mode if self.enable_visual_adapter else "learnable"
            ),
            "use_rms_norm": self.use_rms_norm,
        }

    def stage1_manifest(self) -> dict[str, Any]:
        """
        训练结束时由 Trainer 写入 final_weights/architecture.json。

        Stage 2 加载前核对此文件和 strict=True 的 Adapter state_dict。
        这里提供待写入内容，不代表已经执行磁盘读写。
        """
        return {
            "ablation_id": self.ablation_id,
            "architecture_fingerprint": self.architecture_fingerprint,
            "architecture_parameters": self.architecture_parameters(),
            "seed": self.seed,
            "run_tag": self.run_tag,
        }

    def assert_compatible_stage1_manifest(self, manifest: dict) -> None:
        """
        拒绝跨消融、跨结构和跨初始化的 Stage 1 checkpoint。

        原则：只继承自己那组 Stage 1，而不是 A3 的权重迁移到
        A3-NoRMS。实际训练路径在 Stage 2 配置中单独推导。
        """
        if not isinstance(manifest, dict):
            raise ValueError("Stage 1 architecture manifest 必须为 dict。")

        if manifest.get("ablation_id") != self.ablation_id:
            raise ValueError(
                "Stage 1 checkpoint 消融组不匹配："
                f"expected={self.ablation_id}, "
                f"found={manifest.get('ablation_id')}"
            )
        if manifest.get("architecture_fingerprint") != self.architecture_fingerprint:
            raise ValueError(
                "Stage 1 checkpoint 架构指纹不匹配，"
                "禁止跨模型权重继承。"
            )

        # 继续比较完整字段，不能只相信文件中自己声明的短指纹。
        if manifest.get("architecture_parameters") != self.architecture_parameters():
            raise ValueError(
                "Stage 1 checkpoint 完整架构配置不匹配。"
            )


@dataclass
class CMDASSMStage2BaseConfig(CMDASSMCommonConfig):
    """
    Stage 2 四个数据集共享的配置。

    Stage 1/2 的 ablation_id 相同，LoRA/Adapter 架构相同；
    Stage 2 允许不同学习率和训练轮次，但必须独立加载本组 Stage 1。
    """

    stage1_seed: int = 2048
    stage1_run_tag: str = "main"

    # Stage 1 的优化器 recipe 独立于本次 Stage 2 的 recipe。
    # 只在此标识 Stage 1 的实验配方名称（不写死具体 LR）。
    # 具体的 LoRA / SSM / Fusion / Lambda / Norm LR，
    # 由 Stage 1 TrainConfig.lr_tag 按实际配置自动生成。
    stage1_lr_recipe_name: str = "CMDA-DLR"

    stage1_mimic_cxr_cache_prefix: str = (
        "mimic_cxr_train_clean_v2_screen160k_seed2048"
    )

    # 4 个 Stage 2 数据集的专属类覆盖该名称。
    dataset_name: str = "unset"
    biomed_condition_text_mode: str = "question"

    # 与原 MoRA-Med Stage 2 的 LR 量级相同，重新匹配新组件。
    num_train_epochs: float = 3.0
    dataloader_num_workers: int = 8
    learning_rate: float = 1e-5
    lora_learning_rate: float = 1e-5
    ssm_learning_rate: float = 3e-5
    visual_projection_learning_rate: float = 3e-5
    cm_radf_learning_rate: float = 2e-5
    lambda_learning_rate: float = 2e-5
    visual_norm_learning_rate: float = 1e-5
    warmup_steps: int = 100
    save_total_limit: int = 20

    stage1_weights_dir: str = field(init=False)
    stage2_experiment_dir: str = field(init=False)
    output_dir: str = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.dataset_name == "unset":
            raise ValueError("Stage 2 必须指定 dataset_name。")

        if self.biomed_condition_text_mode != "question":
            raise ValueError(
                "Stage 2 BioMedCLIP 只能读取问题，不允许目标答案。"
            )

        if isinstance(self.stage1_seed, bool) or not (
            isinstance(self.stage1_seed, int) and self.stage1_seed >= 0
        ):
            raise ValueError("stage1_seed 必须为非负整数。")

        validate_run_tag(self.stage1_run_tag, "stage1_run_tag")

        validate_run_tag(
            self.stage1_lr_recipe_name,
            "stage1_lr_recipe_name",
        )

        # Stage 1 的分组 LR 与 Stage 2 的分组 LR 本来就不同。
        # 因此不能在 Stage 2 用 self.lr_tag 推算 Stage 1 路径，
        # 更不能维护写死的 Stage 1 LR 数字字符串。
        #
        # 直接构造相同消融组的 Stage 1 配置，得到唯一可信的
        # Stage 1 final_weights 路径；这一导入位于方法内，
        # 不会造成配置模块加载阶段的循环导入。
        from config.stage1_train_config_mimic_cxr import (
            TrainConfig as Stage1TrainConfig,
        )

        stage1_cfg = Stage1TrainConfig(
            ablation_id=self.ablation_id,
            seed=self.stage1_seed,
            run_tag=self.stage1_run_tag,
            lr_recipe_name=self.stage1_lr_recipe_name,
            output_root=self.output_root,
            mimic_cxr_cache_prefix=self.stage1_mimic_cxr_cache_prefix,
        )

        # Stage 1 / Stage 2 的网络结构与初始化参数必须匹配，
        # 禁止同一消融名下使用不兼容的 Stage 1 Adapter。
        if self.architecture_fingerprint != stage1_cfg.architecture_fingerprint:
            raise ValueError(
                "Stage 2 架构配置与 Stage 1 不一致，"
                "无法安全继承 Stage 1 权重。"
            )

        self.stage1_weights_dir = stage1_cfg.final_weights_dir

        self.stage2_experiment_dir = os.path.join(
            self.output_root,
            f"Stage2_{self.dataset_name}",
            self.architecture_tag,
            f"from_Stage1_seed_{self.stage1_seed}",
            self.lr_tag,
            f"seed_{self.seed}",
            f"run_{self.run_tag}",
        )
        self.output_dir = self.stage2_experiment_dir

    def assert_compatible_stage1_manifest(self, manifest: dict) -> None:
        """额外核验 Stage 1 来源 seed/run_tag，避免取错同组训练。"""
        super().assert_compatible_stage1_manifest(manifest)

        if manifest.get("seed") != self.stage1_seed:
            raise ValueError("Stage 1 checkpoint 的随机种子不匹配。")

        if manifest.get("run_tag") != self.stage1_run_tag:
            raise ValueError("Stage 1 checkpoint 的 run_tag 不匹配。")