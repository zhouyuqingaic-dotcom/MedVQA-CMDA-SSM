"""
CM-DA-SSM · Stage 1 Checkpoint 工具

本文件提供三个与具体 Trainer 无关的核心能力：
    1. Stage 1 最终/中途 checkpoint 的视觉 Adapter 和 manifest 保存；
    2. 安全的 checkpoint 选择、架构校验与 strict=True Adapter 恢复；
    3. 通过 Hugging Face TrainerCallback 扩展每次 checkpoint 保存。

为什么不能只依赖 trainer.save_model()？
    训练模型是 PEFT PeftModel，save_model 默认主要保存 LoRA。
    visual.res_adapter 不是 LoRA，因此必须另外保存 visual_adapter.pt。
    否则断点恢复可能只恢复 LoRA 和优化器状态，却从随机值重新
    初始化 CM-DA-SSM，导致训练状态错位。

本模块不加载 Qwen3-VL-8B 或 BiomedCLIP。易于在 CPU 上独立测试。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ARCHITECTURE_FILE = "architecture.json"
VISUAL_ADAPTER_FILE = "visual_adapter.pt"
FORMAT_VERSION = "CM-DA-SSM-STAGE1-V1"


def fingerprint_train_dataset(dataset) -> str:
    """
    对本次 Stage 1 实际使用的样本顺序生成确定性指纹。

    MIMICCXRDataset.samples 是过滤后的 CSV 行字典列表。
    通过逐条增量 SHA256 既能确认 160k 子集没有变化，
    又不用把整个 JSON 数组复制到内存中。

    不包含报告目标内容本身，因此不会把 IMPRESSION 写入 manifest。
    """
    rows = getattr(dataset, "samples", None)
    if rows is None:
        raise ValueError(
            "需要访问 MIMICCXRDataset.samples 生成样本指纹，"
            "不能只用样本数量判断训练数据完全一致。"
        )

    digest = hashlib.sha256()

    for row in rows:
        canonical = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\\n")

    return digest.hexdigest()


def is_global_main_process(trainer) -> bool:
    """使用 HF Trainer 的全局主进程判断，而非 LOCAL_RANK==0。"""
    return bool(trainer.is_world_process_zero())


def get_visual_adapter(model: nn.Module) -> nn.Module | None:
    """
    找到唯一的 *.visual.res_adapter，避免使用 DDP/PEFT 的固定前缀。

    允许 model.module 包装，但拒绝发现多个视觉适配器的情况。
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model 必须是 nn.Module。")

    adapters = [
        module
        for name, module in model.named_modules()
        if name.endswith(".visual.res_adapter")
    ]

    if len(adapters) > 1:
        raise RuntimeError("模型含多个 visual.res_adapter，无法安全保存。")

    return adapters[0] if adapters else None


def _sha256_file(path: Path) -> str:
    """以流式方式计算哈希，避免将 LoRA 文件整体载入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _find_peft_weights(directory: Path) -> Path:
    """
    检查 Trainer / PEFT 是否已经真正写出了 LoRA 权重。

    当前工程使用默认名称的 PEFT Adapter，权重应位于目标目录。
    不将 PyTorch 全模型快照误认作 LoRA 文件。
    """
    candidates = [
        directory / "adapter_model.safetensors",
        directory / "adapter_model.bin",
    ]
    existing = [candidate for candidate in candidates if candidate.is_file()]

    if len(existing) != 1:
        raise RuntimeError(
            "LoRA 权重保存不完整或不唯一："
            f"目录={directory}，存在文件={[str(p.name) for p in existing]}。"
            "请确认使用默认 PEFT adapter 并已成功执行 trainer.save_model()。"
        )

    if not (directory / "adapter_config.json").is_file():
        raise RuntimeError(
            "没有找到 PEFT adapter_config.json，"
            f"不能把 {directory} 视为完整 LoRA checkpoint。"
        )

    return existing[0]


def _write_json_atomic(path: Path, payload: dict) -> None:
    """manifest 最后写入；只有完全保存成功的目录才有它。"""
    temp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}"
    )
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_torch_atomic(path: Path, state_dict: dict) -> None:
    """防止程序被杀时将半写入的 visual_adapter.pt 作为合法文件。"""
    temp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}"
    )
    try:
        torch.save(state_dict, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _cpu_adapter_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """
    将 Adapter state_dict 整体转换到 CPU（包含固定 λ 等 buffer）。

    只保存当前模块，不碰 Qwen backbone、冻结 BiomedCLIP 或其他组权重。
    """
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in module.state_dict().items()
    }


def _stage1_training_protocol(cfg) -> dict[str, Any]:
    """
    描述 Stage 1 训练中的关键预算与数据处理配置。

    这些字段只用来校验断点恢复。Stage 2 继承网络参数时
    可以使用与 Stage 1 不同的学习率/轮次。
    """
    names = (
        "model_name_or_path",
        "mimic_cxr_max_size",
        "mimic_cxr_target_section",
        "mimic_cxr_instruction_suffix",
        "max_mimic_cxr_train_samples",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "max_steps",
        "warmup_steps",
        "lr_scheduler_type",
        "weight_decay",
        "lora_learning_rate",
        "ssm_learning_rate",
        "visual_projection_learning_rate",
        "cm_radf_learning_rate",
        "lambda_learning_rate",
        "visual_norm_learning_rate",
        "max_grad_norm",
        "torch_dtype",
        "load_in_4bit",
    )

    values = {name: getattr(cfg, name) for name in names}
    values["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    return values


def save_stage1_artifacts(
    *,
    model: nn.Module,
    cfg,
    directory: str | Path,
    global_step: int,
    train_sample_count: int,
    is_final: bool,
    train_sample_digest: str | None = None,
) -> dict[str, Any]:
    """
    保存视觉 Adapter 和架构清单，作为 LoRA 保存后的第二阶段操作。

    前置条件：
        caller 已调用 trainer.save_model(...) 或 HF on_save，
        PEFT 权重和 adapter_config.json 已存在。

    保存后的目录：
        adapter_config.json
        adapter_model.safetensors （或 adapter_model.bin）
        visual_adapter.pt        （A1/A2/A3/A3-NoRMS 有，A0 无）
        architecture.json

    manifest 会保留 cfg.stage1_manifest() 的所有字段，使 Stage 2
    能调用 cfg.assert_compatible_stage1_manifest(manifest) 核验。
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"训练权重目录不存在：{directory}")

    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step 必须为非负整数。")

    if not isinstance(train_sample_count, int) or train_sample_count <= 0:
        raise ValueError("train_sample_count 必须为正整数。")

    lora_path = _find_peft_weights(directory)

    adapter = get_visual_adapter(model)

    if cfg.enable_visual_adapter and adapter is None:
        raise RuntimeError(
            f"{cfg.ablation_id} 启用视觉分支，但模型中没有 visual.res_adapter。"
        )
    if not cfg.enable_visual_adapter and adapter is not None:
        raise RuntimeError("A0 不能保存视觉 Adapter 参数。")

    lora_info = {
        "name": lora_path.name,
        "sha256": _sha256_file(lora_path),
        "bytes": lora_path.stat().st_size,

        # LoRA 权重与 PEFT 结构配置是一套整体。
        # 仅校验权重文件不能发现 adapter_config.json 被替换的问题。
        "config_sha256": _sha256_file(
            directory / "adapter_config.json"
        ),
    }

    adapter_info = None

    if adapter is not None:
        # 先复制到 CPU，再执行原子保存，不改变模型训练状态。
        state = _cpu_adapter_state(adapter)

        adapter_path = directory / VISUAL_ADAPTER_FILE
        _write_torch_atomic(adapter_path, state)

        adapter_info = {
            "name": VISUAL_ADAPTER_FILE,
            "sha256": _sha256_file(adapter_path),
            "tensor_count": len(state),
            "bytes": adapter_path.stat().st_size,
        }
    else:
        # A0 的目录内不能混入其他消融遗留的 Adapter 文件。
        if (directory / VISUAL_ADAPTER_FILE).exists():
            raise RuntimeError(
                "A0 目录中意外存在 visual_adapter.pt，请检查是否复用了其他组目录。"
            )

    manifest = cfg.stage1_manifest()

    # 与 Stage 2 一致的原有字段：ablation_id / fingerprint /
    # architecture_parameters / seed / run_tag 保持不变。
    manifest.update(
        {
            "format_version": FORMAT_VERSION,
            "stage": "Stage1",
            "dataset": "MIMIC-CXR",
            "dataset_cache_prefix": cfg.mimic_cxr_cache_prefix,
            "train_sample_count": train_sample_count,
            "train_sample_digest": train_sample_digest,
            "training_protocol": _stage1_training_protocol(cfg),
            "global_step": global_step,
            "is_final": bool(is_final),
            "artifacts": {
                "lora": lora_info,
                "visual_adapter": adapter_info,
            },
        }
    )

    # 只在 LoRA 和 visual_adapter 完整存在、哈希计算成功后写 manifest。
    _write_json_atomic(directory / ARCHITECTURE_FILE, manifest)

    return manifest


def read_stage1_manifest(directory: str | Path) -> dict[str, Any]:
    """读取 architecture.json，不执行任何非信任代码。"""
    path = Path(directory) / ARCHITECTURE_FILE

    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 Stage 1 架构清单：{path}。"
            "这是不完整/旧版 checkpoint，禁止直接恢复。"
        )

    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    if not isinstance(result, dict):
        raise ValueError("architecture.json 内容必须是 JSON 对象。")

    return result


def validate_stage1_artifacts(
    *,
    cfg,
    directory: str | Path,
    expected_train_sample_count: int | None = None,
    expected_train_sample_digest: str | None = None,
    require_optimizer_state: bool = False,
) -> dict[str, Any]:
    """
    恢复/继承权重前严格核验，不允许跨组、跨 seed 载入。

    require_optimizer_state=True 用于断点续训：
        必须含 Trainer state、optimizer、scheduler 和 RNG 文件。
    Stage 2 仅继承最终 LoRA + Adapter，不要求这些状态文件。

    本函数只校验文件及哈希，不实例化新的 GPU 模型。
    """
    directory = Path(directory)
    manifest = read_stage1_manifest(directory)

    cfg.assert_compatible_stage1_manifest(manifest)

    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Stage 1 checkpoint 格式版本不匹配。"
        )

    if manifest.get("stage") != "Stage1":
        raise ValueError("此目录不是 Stage 1 权重。")

    if manifest.get("dataset") != "MIMIC-CXR":
        raise ValueError("Stage 1 数据集不是 MIMIC-CXR。")

    if manifest.get("seed") != cfg.seed:
        raise ValueError("Stage 1 checkpoint seed 不一致。")

    if manifest.get("run_tag") != cfg.run_tag:
        raise ValueError("Stage 1 checkpoint run_tag 不一致。")

    if manifest.get("dataset_cache_prefix") != cfg.mimic_cxr_cache_prefix:
        raise ValueError(
            "Stage 1 checkpoint 数据集缓存前缀与当前配置不一致。"
        )

    if expected_train_sample_count is not None:
        if manifest.get("train_sample_count") != expected_train_sample_count:
            raise ValueError(
                "训练集样本数与 checkpoint 不一致，"
                "不能精确续训。"
            )

    if expected_train_sample_digest is not None:
        if manifest.get("train_sample_digest") != expected_train_sample_digest:
            raise ValueError(
                "Stage 1 的训练样本顺序/内容与 checkpoint 不一致，"
                "即使样本数量相同也不能精确续训。"
            )

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("checkpoint artifacts 字段非法。")

    lora_info = artifacts.get("lora")
    if not isinstance(lora_info, dict):
        raise ValueError("checkpoint 未记录 LoRA 文件。")

    lora_file = _find_peft_weights(directory)
    if lora_info.get("name") != lora_file.name:
        raise ValueError("LoRA 权重文件名称与 manifest 不一致。")

    if lora_info.get("sha256") != _sha256_file(lora_file):
        raise ValueError("LoRA 权重哈希校验失败，文件可能损坏。")

    if lora_info.get("config_sha256") != _sha256_file(
        directory / "adapter_config.json"
    ):
        raise ValueError(
            "LoRA adapter_config.json 哈希校验失败。"
        )

    adapter_info = artifacts.get("visual_adapter")
    adapter_file = directory / VISUAL_ADAPTER_FILE

    if cfg.enable_visual_adapter:
        if not isinstance(adapter_info, dict) or not adapter_file.is_file():
            raise RuntimeError(
                f"{cfg.ablation_id} checkpoint 缺少 visual_adapter.pt。"
            )
        if adapter_info.get("name") != VISUAL_ADAPTER_FILE:
            raise ValueError("视觉 Adapter 文件名与 manifest 不一致。")
        if adapter_info.get("sha256") != _sha256_file(adapter_file):
            raise ValueError("视觉 Adapter 权重哈希校验失败。")
    else:
        if adapter_info is not None or adapter_file.exists():
            raise RuntimeError("A0 checkpoint 不应包含 visual_adapter.pt。")

    if require_optimizer_state:
        if manifest.get("training_protocol") != _stage1_training_protocol(cfg):
            raise ValueError(
                "Stage 1 训练协议发生变化（Batch/LR/模型精度/World Size 等），"
                "无法保证精确断点续训，请恢复原配置或使用新的 run_tag。"
            )

        required = (
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
        )
        missing = [
            name
            for name in required
            if not (directory / name).is_file()
        ]

        if not list(directory.glob("rng_state*.pth")):
            missing.append("rng_state*.pth")

        if missing:
            raise RuntimeError(
                "断点续训缺少 Trainer 状态文件："
                f"{missing}。"
                "不能只从 LoRA 权重重新开始却声称是精确恢复。"
            )

    return manifest


def restore_visual_adapter_strict(
    model: nn.Module,
    *,
    cfg,
    directory: str | Path,
) -> None:
    """
    只恢复视觉 Adapter，并要求 strict=True。

    Trainer.train(resume_from_checkpoint=...) 负责加载 LoRA、
    optimizer/scheduler/RNG。本函数保证视觉 Adapter 在其之前恢复。
    """
    directory = Path(directory)
    adapter = get_visual_adapter(model)

    if not cfg.enable_visual_adapter:
        if adapter is not None:
            raise RuntimeError("A0 不应有视觉 Adapter。")
        return

    if adapter is None:
        raise RuntimeError("恢复视觉权重失败：模型没有 visual.res_adapter。")

    path = directory / VISUAL_ADAPTER_FILE

    if not path.is_file():
        raise FileNotFoundError(f"缺少视觉 Adapter 权重：{path}")

    # 只允许 tensor state_dict，避免 torch.load 执行任意 pickled 代码。
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, dict):
        raise ValueError("visual_adapter.pt 必须是 state_dict 字典。")

    # key/shape/dtype 的结构不一致则立即停止训练。
    adapter.load_state_dict(state, strict=True)


def _checkpoint_step(path: Path) -> int | None:
    """仅识别 checkpoint-<非负整数>，不把临时目录当成 checkpoint。"""
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match and path.is_dir():
        return int(match.group(1))
    return None


def resolve_resume_checkpoint(
    *,
    cfg,
    output_dir: str | Path,
    requested: str | None,
    expected_train_sample_count: int | None = None,
    expected_train_sample_digest: str | None = None,
) -> str | None:
    """
    将 CLI 的 --resume_from_checkpoint 解析为真实 checkpoint 路径。

    允许：
        requested=None：开始全新训练（如有历史 checkpoint 则拒绝覆盖）。
        requested="auto"：从本实验最后一个有效完整 checkpoint 继续。
        requested=".../checkpoint-500"：从本实验指定 checkpoint 继续。

    安全：
        只能从当前消融/seed/run_tag 的 output_dir/checkpoint-N 续训；
        不允许指向其他组、甚至其他随机种子的 checkpoint。
    """
    output_dir = Path(output_dir).resolve()
    children = list(output_dir.iterdir()) if output_dir.exists() else []
    checkpoints = [
        (step, p)
        for p in children
        if (step := _checkpoint_step(p)) is not None
    ]

    checkpoints.sort(key=lambda pair: pair[0], reverse=True)

    final_path = output_dir / "final_weights"

    if requested is None or requested == "none":
        if checkpoints or (
            final_path.exists()
            and (
                not final_path.is_dir()
                or any(final_path.iterdir())
            )
        ):
            raise RuntimeError(
                "当前输出目录已有 checkpoint 或 final_weights。"
                "为了不覆盖之前的训练，请明确指定 "
                "--resume_from_checkpoint auto，或更换 --run_tag。"
            )
        return None

    if requested == "auto":
        if not checkpoints:
            raise FileNotFoundError(
                "auto resume 失败：当前实验目录没有 checkpoint-N。"
            )

        failures = []

        for _, candidate in checkpoints:
            try:
                validate_stage1_artifacts(
                    cfg=cfg,
                    directory=candidate,
                    expected_train_sample_count=expected_train_sample_count,
                    expected_train_sample_digest=expected_train_sample_digest,
                    require_optimizer_state=True,
                )
                return str(candidate)
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                failures.append(f"{candidate.name}: {exc}")

        raise RuntimeError(
            "未找到可精确恢复的 Stage 1 checkpoint：\n"
            + "\n".join(failures)
        )

    checkpoint = Path(requested).expanduser().resolve()

    # 拒绝任意外部路径与跨消融目录，即使 manifest 恰好可读取。
    if checkpoint.parent != output_dir or _checkpoint_step(checkpoint) is None:
        raise ValueError(
            "只能从本次 Stage 1 output_dir 内的 checkpoint-N 恢复，"
            f"实际收到：{checkpoint}"
        )

    validate_stage1_artifacts(
        cfg=cfg,
        directory=checkpoint,
        expected_train_sample_count=expected_train_sample_count,
        expected_train_sample_digest=expected_train_sample_digest,
        require_optimizer_state=True,
    )

    return str(checkpoint)


def create_stage1_checkpoint_callback(
    cfg,
    train_sample_count: int,
    train_sample_digest: str | None = None,
):
    """
    延迟导入 Transformers，并创建真实 TrainerCallback。

    on_save 在 Trainer 自己保存 LoRA/optimizer/scheduler/RNG 后执行，
    因此此时可以安全附加 visual_adapter.pt + architecture.json。

    不能只在训练完成后保存视觉 Adapter，否则断点恢复会不完整。
    """
    from transformers import TrainerCallback

    class Stage1ExtraCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control

            checkpoint_dir = (
                Path(args.output_dir)
                / f"checkpoint-{state.global_step}"
            )

            model = kwargs.get("model")
            if model is None:
                raise RuntimeError(
                    "TrainerCallback 未传入 model，"
                    "无法保存视觉 Adapter。"
                )

            save_stage1_artifacts(
                model=model,
                cfg=cfg,
                directory=checkpoint_dir,
                global_step=int(state.global_step),
                train_sample_count=train_sample_count,
                train_sample_digest=train_sample_digest,
                is_final=False,
            )

            return control

    return Stage1ExtraCheckpointCallback()
