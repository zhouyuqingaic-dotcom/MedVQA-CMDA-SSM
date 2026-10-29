"""
CM-DA-SSM 五组消融使用的 Discriminative AdamW。

不再保留旧 MoRA-Med 的 F3/F5/F7 卷积专家和三专家 Router/Gate 分类。

A0:
    只训练 LoRA 参数（没有视觉适配器）。

A1:
    LoRA + 四方向 SSM + Shared Down/Up + Norm。
    融合为严格 1/4 均匀权重，不存在 Query/Key 的可训练参数，
    lambda 为固定 buffer，不加入优化器。

A2:
    A1 的 SSM + Shared Down/Up/Norm + CM-RADF Query/Key，
    lambda 仍为固定 buffer。

A3:
    A2 + 可学习 lambda_a，Capped RMS。

A3-NoRMS:
    与 A3 相同的可训练参数结构，惟 rho=1。
    必须从自己的 Stage 1 权重继续训练，不能读取 A3 的 Stage 1。

参数不漏训，冻结层不误训，weight decay 排除规则详见分组函数。
"""

from collections import defaultdict
import math

import torch
import torch.nn as nn

from config.cm_da_ssm.ablation_config import ABLATION_IDS, ABLATION_RULES


# 按顺序创建参数组：与 Hugging Face Trainer 的首组 LR 日志保持一致。
COMPONENT_ORDER = (
    "lora",
    "ssm",
    "visual_projection",
    "cm_radf",
    "lambda",
    "visual_norm",
)

SSM_DIRECTION_NAMES = (
    "ssm_rf.",
    "ssm_rr.",
    "ssm_cf.",
    "ssm_cr.",
)

SSM_PREFIX = "feature_adapter.four_directional_ssm."
FUSION_PREFIX = "feature_adapter.cm_radf."


def _locate_visual_adapter(model: nn.Module) -> nn.Module | None:
    """
    按 Step 3 的实际挂载路径定位 visual.res_adapter。

    不依赖 PEFT 外层的固定层数：同样支持 model.module
    和 base_model.model.model.visual.res_adapter 的参数命名。

    严格要求最多一个 Adapter，防止多模型意外混用。
    """
    found = [
        module
        for name, module in model.named_modules()
        if name == "res_adapter" or name.endswith(".res_adapter")
    ]

    if len(found) > 1:
        raise RuntimeError(
            "模型里找到多个 res_adapter，无法确定唯一视觉适配器。"
        )

    return found[0] if found else None


def _classify_parameter(
    name: str,
    parameter: nn.Parameter,
    norm_ids: set[int],
) -> tuple[str, bool] | None:
    """
    返回 (component, no_decay)，无法分类则返回 None。

    1. LoRA 是独立的第一组；
    2. 其他参数必须存在于 *.res_adapter.* 内；
    3. LayerNorm 被归入 visual_norm，不受父模块名字影响；
    4. Mamba A_log / D 与特殊嵌入等参数不做 weight decay。
    """
    if "lora_" in name and "res_adapter" not in name:
        no_decay = name.endswith(".bias")
        return "lora", no_decay

    if ".res_adapter." in name:
        relative = name.split(".res_adapter.", 1)[1]
    elif name.startswith("res_adapter."):
        relative = name[len("res_adapter."):]
    else:
        # Qwen 基座/BioMedCLIP 如有任何意外解冻参数，应直接失败。
        return None

    # 全局 lambda 只有 A3 和 A3-NoRMS 才可学习。
    if relative == "lambda_a":
        component = "lambda"

    elif relative.startswith(SSM_PREFIX):
        local = relative[len(SSM_PREFIX):]

        if local.startswith(SSM_DIRECTION_NAMES):
            component = "ssm"
        elif local.startswith("down."):
            component = "visual_projection"
        elif local.startswith(("ln_in.", "ln_dir.")):
            component = "visual_norm"
        else:
            return None

    elif relative.startswith(FUSION_PREFIX):
        local = relative[len(FUSION_PREFIX):]

        if local.startswith("up."):
            component = "visual_projection"
        elif local.startswith("ln_fuse."):
            component = "visual_norm"
        elif (
            local.startswith(
                (
                    "router_backbone.",
                    "query_proj.",
                    "global_key_proj.",
                    "local_key_proj.",
                )
            )
            or local in {"direction_embeddings", "raw_eta"}
        ):
            component = "cm_radf"
        else:
            return None

    else:
        return None

    # 将 Condition MLP 内部 LN 也归一到 Norm 组。
    if id(parameter) in norm_ids:
        component = "visual_norm"

    # 不衰减 Norm、bias、Mamba 状态矩阵、方向嵌入、lambda、eta。
    no_decay = (
        id(parameter) in norm_ids
        or name.endswith(".bias")
        or relative in {"lambda_a"}
        or relative.endswith(
            (
                ".A_log",
                ".D",
                ".direction_embeddings",
                ".raw_eta",
            )
        )
        or relative in {"direction_embeddings", "raw_eta"}
        or bool(getattr(parameter, "_no_weight_decay", False))
    )

    return component, no_decay


def _validate_ablation_adapter(
    ablation_id: str,
    adapter: nn.Module | None,
) -> None:
    """
    检查真实实例化的模型与消融配置相符。

    不允许只将文件夹命名为 A1，却实际训练 CM-RADF；
    不允许将 A3-NoRMS 命名后却仍使用 Capped RMS；
    不允许 A2 偷偷学习 lambda。
    """
    rule = ABLATION_RULES[ablation_id]

    if not rule["enable_visual_adapter"]:
        if adapter is not None:
            raise RuntimeError(
                "A0 只能训练 LoRA，不能挂载 res_adapter。"
            )
        return

    if adapter is None:
        raise RuntimeError(
            f"{ablation_id} 需要 res_adapter，但模型尚未挂载。"
        )

    for attr in ("fusion_mode", "lambda_mode", "use_rms_norm"):
        expected = rule[attr]
        actual = getattr(adapter, attr, None)

        if actual != expected:
            raise RuntimeError(
                f"{ablation_id} 的 {attr} 不匹配："
                f"expected={expected!r}, actual={actual!r}"
            )

    # A1 的 Uniform 模块不应创建 CM-RADF Query/Key。
    # A2+ 则必须拥有实际 GlobalLocalCMRADF。
    if not hasattr(adapter, "feature_adapter"):
        raise RuntimeError("res_adapter 缺少 feature_adapter。")

    fusion = adapter.feature_adapter.cm_radf

    if ablation_id == "A1":
        forbidden = (
            "query_proj",
            "router_backbone",
            "global_key_proj",
            "local_key_proj",
            "direction_embeddings",
        )
        if any(hasattr(fusion, name) for name in forbidden):
            raise RuntimeError(
                "A1 Uniform 分支不能创建 CM-RADF Query/Key 参数。"
            )

    if ablation_id in {"A2", "A3", "A3-NoRMS"}:
        required = (
            "query_proj",
            "global_key_proj",
            "local_key_proj",
            "direction_embeddings",
        )
        if any(not hasattr(fusion, name) for name in required):
            raise RuntimeError(
                f"{ablation_id} 必须使用完整 CM-RADF。"
            )

    # 训练可学习与固定 lambda 不只靠 requires_grad 区分。
    # 固定 lambda 没有 lambda_a Parameter，仅保留 fixed_lambda buffer。
    lambda_parameter = getattr(adapter, "lambda_a", None)

    if rule["lambda_mode"] == "fixed":
        if lambda_parameter is not None:
            raise RuntimeError(
                f"{ablation_id} 固定 λ 时不允许存在 lambda_a 参数。"
            )
        if getattr(adapter, "fixed_lambda", None) is None:
            raise RuntimeError(
                f"{ablation_id} 缺少固定 λ buffer。"
            )

    else:
        if not isinstance(lambda_parameter, nn.Parameter):
            raise RuntimeError(
                f"{ablation_id} 必须包含可学习 lambda_a 参数。"
            )
        if not lambda_parameter.requires_grad:
            raise RuntimeError(
                f"{ablation_id} 的 lambda_a 必须是可训练的。"
            )


def build_discriminative_adamw(
    model: nn.Module,
    *,
    ablation_id: str,
    lora_learning_rate: float,
    ssm_learning_rate: float,
    visual_projection_learning_rate: float,
    cm_radf_learning_rate: float,
    lambda_learning_rate: float,
    visual_norm_learning_rate: float,
    weight_decay: float,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
) -> torch.optim.AdamW:
    """
    新 CM-DA-SSM 的唯一分组优化器入口。

    使用 cfg.optimizer_kwargs(training_args) 直接构造，避免
    手工传入旧 visual_expert_learning_rate/router_gate_learning_rate。

    根据 ablation_id 自动要求正确的参数组：
        A0        -> lora
        A1        -> lora/ssm/projection/norm
        A2        -> A1 + cm_radf
        A3        -> A2 + lambda
        A3-NoRMS  -> 与 A3 完全相同的参数组

    注意 A3 与 A3-NoRMS 的区别只在 Residual Adapter 的 rho 公式，
    不在 optimizer 组别、初始化和学习率。
    """
    if ablation_id not in ABLATION_IDS:
        raise ValueError(
            f"未知消融实验 {ablation_id!r}，只支持 {ABLATION_IDS}"
        )

    rates = {
        "lora": float(lora_learning_rate),
        "ssm": float(ssm_learning_rate),
        "visual_projection": float(visual_projection_learning_rate),
        "cm_radf": float(cm_radf_learning_rate),
        "lambda": float(lambda_learning_rate),
        "visual_norm": float(visual_norm_learning_rate),
    }

    for name, lr in rates.items():
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(
                f"{name} 的 learning rate 必须为正有限数，实际为 {lr}"
            )

    if not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("weight_decay 必须为非负有限数。")

    if not (0 <= adam_beta1 < 1 and 0 <= adam_beta2 < 1):
        raise ValueError("Adam beta1/beta2 必须在 [0,1)。")

    if not math.isfinite(float(adam_epsilon)) or adam_epsilon <= 0:
        raise ValueError("Adam epsilon 必须为正有限数。")

    adapter = _locate_visual_adapter(model)
    _validate_ablation_adapter(ablation_id, adapter)

    # 精确识别所有 LN（包括 router_backbone 内 LayerNorm）。
    norm_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }

    buckets: dict[tuple[str, bool], list[nn.Parameter]] = defaultdict(list)
    unmatched = []
    seen_ids = set()
    trainable_names = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        trainable_names.add(name)
        result = _classify_parameter(name, parameter, norm_ids)

        if result is None:
            unmatched.append(name)
            continue

        component, no_decay = result
        param_id = id(parameter)

        if param_id in seen_ids:
            raise RuntimeError(f"参数重复分组：{name}")

        seen_ids.add(param_id)
        buckets[(component, no_decay)].append(parameter)

    if unmatched:
        message = "\n".join(f"  - {n}" for n in unmatched[:50])
        raise RuntimeError(
            "发现无法归入 CM-DA-SSM 优化器的可训练参数：\n"
            f"{message}\n"
            "请检查是否意外解冻了 Qwen/BioMedCLIP，"
            "或新增了未配置的 Adapter 模块。"
        )

    expected = {"lora"}
    if ablation_id != "A0":
        expected.update(
            {"ssm", "visual_projection", "visual_norm"}
        )
        for direction in SSM_DIRECTION_NAMES:
            if not any(
                SSM_PREFIX + direction in name
                for name in trainable_names
            ):
                raise RuntimeError(
                    f"缺少可训练 SSM 方向分支：{direction}"
                )

    if ablation_id in {"A2", "A3", "A3-NoRMS"}:
        expected.add("cm_radf")

    if ablation_id in {"A3", "A3-NoRMS"}:
        expected.add("lambda")

    present = {part for part, no_decay in buckets}

    if present != expected:
        raise RuntimeError(
            f"{ablation_id} 参数组不符合消融计划："
            f"expected={sorted(expected)}, actual={sorted(present)}"
        )

    # A1/A2 两组固定 λ，不得出现 trainable lambda_a。
    # A3/A3-NoRMS 则必须出现且共享学习率。
    if ablation_id in {"A1", "A2"}:
        if any(name.endswith(".lambda_a") for name in trainable_names):
            raise RuntimeError("A1/A2 不允许训练 lambda_a。")

    groups = []

    for component in COMPONENT_ORDER:
        for no_decay in (False, True):
            parameters = buckets.get(
                (component, no_decay),
                [],
            )
            if not parameters:
                continue

            groups.append(
                {
                    "params": parameters,
                    "lr": rates[component],
                    "weight_decay": (
                        0.0 if no_decay else float(weight_decay)
                    ),
                    "group_name": (
                        f"{component}/no_decay"
                        if no_decay else f"{component}/decay"
                    ),
                }
            )

    if not groups:
        raise RuntimeError("优化器没有找到任何可训练参数。")

    # 所有 trainable 参数必须完整覆盖、不重复。
    expected_ids = {
        id(p)
        for p in model.parameters()
        if p.requires_grad
    }
    assigned_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]

    if (
        len(assigned_ids) != len(set(assigned_ids))
        or set(assigned_ids) != expected_ids
    ):
        raise RuntimeError(
            "优化器参数去重或完整覆盖检查失败。"
        )

    return torch.optim.AdamW(
        groups,
        betas=(float(adam_beta1), float(adam_beta2)),
        eps=float(adam_epsilon),
    )


def format_optimizer_groups(
    optimizer: torch.optim.Optimizer,
) -> str:
    """
    打印各参数组学习率、是否衰减和参数数量。

    HF Trainer 通常只显示首组 LoRA 的 LR；
    正式训练建议额外打印本函数确保全部组设置正确。
    """
    lines = ["CM-DA-SSM Discriminative AdamW parameter groups:"]

    for group in optimizer.param_groups:
        parameter_count = sum(
            p.numel()
            for p in group["params"]
        )
        lines.append(
            "  - "
            f"{group.get('group_name', 'unnamed')}: "
            f"lr={group['lr']:.3e}, "
            f"weight_decay={group['weight_decay']:.3g}, "
            f"params={parameter_count:,}"
        )

    return "\n".join(lines)
