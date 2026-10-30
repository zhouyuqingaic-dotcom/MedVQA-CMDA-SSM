"""
CM-DA-SSM Stage 2 · SLAKE 训练入口（五组消融统一）。

必须与 Stage 1 保持一致：
    A0        Qwen3-VL + LoRA
    A1        4D-SSM + Uniform Fusion + Fixed λ + Capped RMS
    A2        4D-SSM + CM-RADF + Fixed λ + Capped RMS
    A3        4D-SSM + CM-RADF + Learnable λ + Capped RMS
    A3-NoRMS  与 A3 相同，ρ≡1

Stage 2 从同一消融组的 Stage 1 final_weights 继承：
    1. PEFT LoRA 权重（adapter_model.safetensors / adapter_model.bin）
    2. CM-DA-SSM 视觉 Adapter（visual_adapter.pt；A0 不需要）

严格禁止：
    - Stage 2 从不匹配的 Stage 1 消融组继承权重；
    - 将旧 MoRA-Med 的三卷积 Expert / Router / Gate 注入新模型；
    - BioMedCLIP 读取标准答案（其文本输入只能是当前问题）；
    - 训练集无意中混入 validate/test 集。

运行示例：
    python training/stage2_trainer_slake.py --check_config_only \\
        --ablation_id A3 --seed 2048 --stage1_seed 2048

    python -m torch.distributed.run --standalone --nproc_per_node=4 \\
        training/stage2_trainer_slake.py --ablation_id A3 \\
        --seed 2048 --stage1_seed 2048 --run_tag main \\
        --stage1_run_tag main

独立冒烟测试（只运行两个 optimizer steps，不发布正式权重）：
    python -m torch.distributed.run --standalone --nproc_per_node=4 \\
        training/stage2_trainer_slake.py --ablation_id A3 --smoke_steps 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset

from config.slake.stage2_train_config_slake import Stage2TrainConfig
from config.stage1_train_config_mimic_cxr import TrainConfig as Stage1TrainConfig
from datas.slake_datasets import SLAKEDataset
from utils.data_tools.prompt_cleaning.slake_answer_cleaning import (
    slake_answer_train_cleaning,
)
from utils.ddp.ddp_utils import ddp_print
from utils.training.discriminative_optimizer import (
    build_discriminative_adamw,
    format_optimizer_groups,
)
from utils.training.stage1_checkpoint_utils import (
    get_visual_adapter,
    restore_visual_adapter_strict,
    validate_stage1_artifacts,
)


def parse_options() -> argparse.Namespace:
    """其余 CLI 参数由 Stage2TrainConfig 解析，避免重复解析冲突。"""
    parser = argparse.ArgumentParser(
        description="CM-DA-SSM Stage 2 SLAKE Trainer"
    )
    parser.add_argument(
        "--check_config_only",
        action="store_true",
        help="不加载 8B 模型、不启动 DDP，检查配置和 Stage 1 权重完整性。",
    )
    parser.add_argument(
        "--check_data_only",
        action="store_true",
        help="在配置检查基础上扫描 SLAKE train 数据，不加载 8B 模型。",
    )
    parser.add_argument(
        "--smoke_steps",
        type=int,
        default=0,
        help="独立运行 N 个优化器步，输出到 _smoke_stepsN，绝不发布正式权重。",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="默认从 Stage 1 重新微调；auto 恢复当前 Stage 2 最近的完整 checkpoint。",
    )
    options, _ = parser.parse_known_args()
    if options.smoke_steps < 0:
        raise ValueError("--smoke_steps 不能为负数。")
    if options.smoke_steps and options.resume_from_checkpoint:
        raise ValueError("smoke_steps 不允许同时使用 Stage 2 断点恢复。")
    return options


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stage1_cfg_for(cfg: Stage2TrainConfig) -> Stage1TrainConfig:
    """Stage 2 仅继承 Stage 1 结构和训练后的参数，不继承优化器。"""
    return Stage1TrainConfig(
        ablation_id=cfg.ablation_id,
        seed=cfg.stage1_seed,
        run_tag=cfg.stage1_run_tag,
        lr_recipe_name=cfg.stage1_lr_recipe_name,
        output_root=cfg.output_root,
        mimic_cxr_cache_prefix=cfg.stage1_mimic_cxr_cache_prefix,
    )


def validate_stage1_for_slake(cfg: Stage2TrainConfig) -> dict:
    """
    用 Stage 1 自身的真实训练配置验证 SHA256、权重和架构；
    再由 Stage 2 校验 seed、run_tag 等继承关系。
    """
    source_cfg = stage1_cfg_for(cfg)
    if Path(source_cfg.final_weights_dir).resolve() != Path(
        cfg.stage1_weights_dir
    ).resolve():
        raise RuntimeError(
            "Stage 2 推导的权重目录与 Stage 1 实际目录不一致。"
        )

    manifest = validate_stage1_artifacts(
        cfg=source_cfg,
        directory=cfg.stage1_weights_dir,
        require_optimizer_state=False,
    )

    cfg.assert_compatible_stage1_manifest(manifest)

    if manifest.get("is_final") is not True:
        raise RuntimeError("Stage 2 必须从已完成的 Stage 1 final_weights 继承。")

    if manifest.get("train_sample_count") != 160000:
        raise RuntimeError(
            "当前 Stage 2 正式实验要求使用 160000 样本的 Stage 1；"
            f"实际为 {manifest.get('train_sample_count')}。"
        )

    if manifest.get("global_step") != 5000:
        raise RuntimeError(
            "当前 Stage 2 正式实验要求 Stage 1 训练 5000 updates；"
            f"实际为 {manifest.get('global_step')}。"
        )

    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_train_data(cfg: Stage2TrainConfig):
    """只读取官方 train.json；validate/test 不得混入训练。"""
    raw = SLAKEDataset(
        json_path=cfg.slake_train_json_path,
        image_root=cfg.slake_image_root,
        verify_images=False,
    )

    valid_indices: list[int] = []
    for index in range(len(raw)):
        sample = raw[index]
        question = str(sample.get("question", "")).strip()
        answer = slake_answer_train_cleaning(
            str(sample.get("answer", "")).strip()
        )
        if question and answer:
            valid_indices.append(index)

    if cfg.max_slake_train_samples is not None:
        if cfg.max_slake_train_samples <= 0:
            raise ValueError("max_slake_train_samples 必须为正整数。")
        valid_indices = valid_indices[:cfg.max_slake_train_samples]

    if not valid_indices:
        raise RuntimeError("SLAKE train 数据集无有效的问答监督样本。")

    ddp_print(
        f"✅ SLAKE Train：原始 {len(raw)} 条，"
        f"有效 {len(valid_indices)} 条，"
        f"过滤 {len(raw) - len(valid_indices)} 条。"
    )
    ddp_print(
        "    本次训练仅使用 train.json；validate.json 和 test.json 不参与训练。"
    )
    return Subset(raw, valid_indices)


def _load_lora_strict(peft_model, stage1_dir: Path) -> None:
    """
    从 Stage 1 加载 LoRA，并比较加载前后每一个训练权重。

    get_peft_model_state_dict 与 PEFT save_pretrained 使用相同键规则；
    对比键/形状和值可避免 load 时全部被意外忽略。
    """
    from peft import (
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )

    path_safe = stage1_dir / "adapter_model.safetensors"
    path_bin = stage1_dir / "adapter_model.bin"
    if path_safe.is_file():
        from safetensors.torch import load_file
        incoming = load_file(str(path_safe), device="cpu")
    elif path_bin.is_file():
        incoming = torch.load(
            path_bin,
            map_location="cpu",
            weights_only=True,
        )
    else:
        raise FileNotFoundError("Stage 1 不包含 LoRA 权重文件。")

    if not isinstance(incoming, dict) or not incoming:
        raise RuntimeError("Stage 1 LoRA state_dict 为空或格式错误。")

    set_peft_model_state_dict(peft_model, incoming)
    restored = get_peft_model_state_dict(peft_model)

    if set(incoming) != set(restored):
        missing = sorted(set(incoming) - set(restored))
        unexpected = sorted(set(restored) - set(incoming))
        raise RuntimeError(
            "Stage 1 LoRA 键与当前 PEFT 模型不一致："
            f"缺少={missing[:5]}, 多余={unexpected[:5]}"
        )

    for key, expected in incoming.items():
        actual = restored[key].detach().cpu()
        # PEFT 可能按目标参数 dtype 恢复（例如 BF16），所以先将
        # checkpoint 值转换为目标 dtype，再做完全相等比较；
        # 避免 FP32 -> BF16 的正常舍入被误判为权重加载失败。
        expected_in_model_dtype = expected.to(dtype=actual.dtype)
        if actual.shape != expected.shape or not torch.equal(
            actual, expected_in_model_dtype
        ):
            raise RuntimeError(
                f"Stage 1 LoRA 权重恢复后数值不一致：{key}"
            )


def build_training_args(cfg, output_dir: str, smoke_steps: int = 0):
    """Stage 2 的学习率、epoch、batch 从 SLAKE 配置读取，不复用 Stage 1。"""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=smoke_steps if smoke_steps > 0 else cfg.max_steps,

        learning_rate=cfg.lora_learning_rate,
        optim="adamw_torch",
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=1 if smoke_steps else cfg.logging_steps,
        save_strategy="no" if smoke_steps else "steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_only_model=False,

        bf16=cfg.torch_dtype in ("bfloat16", torch.bfloat16),
        fp16=cfg.torch_dtype in ("float16", torch.float16),
        dataloader_num_workers=cfg.dataloader_num_workers,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
    )


def _stage2_manifest(cfg, stage1_manifest: dict, global_step: int, train_count: int):
    """保存 Stage 2 对应的 Stage 1 来源和实际超参数，便于将来评测。"""
    return {
        "format_version": "CM-DA-SSM-STAGE2-SLAKE-V1",
        "dataset": "SLAKE",
        "stage": "Stage2",
        "ablation_id": cfg.ablation_id,
        "architecture_fingerprint": cfg.architecture_fingerprint,
        "architecture_parameters": cfg.architecture_parameters(),
        "stage1_weights_dir": cfg.stage1_weights_dir,
        "stage1_global_step": stage1_manifest.get("global_step"),
        "stage1_architecture_fingerprint": stage1_manifest.get(
            "architecture_fingerprint"
        ),
        "seed": cfg.seed,
        "stage1_seed": cfg.stage1_seed,
        "run_tag": cfg.run_tag,
        "stage1_run_tag": cfg.stage1_run_tag,
        "global_step": int(global_step),
        "train_sample_count": int(train_count),
        "training_settings": {
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "num_train_epochs": cfg.num_train_epochs,
            "lora_learning_rate": cfg.lora_learning_rate,
            "ssm_learning_rate": cfg.ssm_learning_rate,
            "visual_projection_learning_rate": cfg.visual_projection_learning_rate,
            "cm_radf_learning_rate": cfg.cm_radf_learning_rate,
            "lambda_learning_rate": cfg.lambda_learning_rate,
            "visual_norm_learning_rate": cfg.visual_norm_learning_rate,
            "warmup_steps": cfg.warmup_steps,
        },
    }


def save_stage2_sidecars(
    *,
    model,
    directory: str | Path,
    cfg,
    stage1_manifest: dict,
    global_step: int,
    train_count: int,
) -> None:
    """
    LoRA 由 Hugging Face Trainer.save_model 保存；
    此函数只保存视觉 Adapter（非 PEFT 权重）和元数据。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    adapter = get_visual_adapter(model)

    if cfg.enable_visual_adapter:
        if adapter is None:
            raise RuntimeError("Stage 2 缺少预期的视觉 Adapter。")
        state = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in adapter.state_dict().items()
        }
        temp = directory / f"visual_adapter.pt.tmp.{os.getpid()}"
        try:
            torch.save(state, temp)
            os.replace(temp, directory / "visual_adapter.pt")
        finally:
            temp.unlink(missing_ok=True)
    elif adapter is not None:
        raise RuntimeError("A0 不应出现视觉 Adapter。")

    metadata = _stage2_manifest(
        cfg, stage1_manifest, global_step, train_count,
    )
    file_adapter = directory / "visual_adapter.pt"
    metadata["visual_adapter_sha256"] = (
        _sha256(file_adapter)
        if cfg.enable_visual_adapter else None
    )
    _write_json_atomic(directory / "stage2_manifest.json", metadata)


def create_checkpoint_callback(cfg, stage1_manifest, train_count):
    """每次 checkpoint 额外保存 Visual Adapter 与 Stage 2 manifest。"""
    from transformers import TrainerCallback

    class SaveStage2Callback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control

            directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            save_stage2_sidecars(
                model=kwargs["model"],
                directory=directory,
                cfg=cfg,
                stage1_manifest=stage1_manifest,
                global_step=state.global_step,
                train_count=train_count,
            )
            return control

    return SaveStage2Callback()


def resolve_resume_checkpoint(cfg, output_dir: str, requested: str | None):
    """
    默认不恢复；如果当前目录已有正式 checkpoint，拒绝覆盖。
    auto 仅使用当前 Stage 2 运行目录中带 manifest 的检查点。
    """
    root = Path(output_dir).resolve()
    checkpoints = sorted(
        (
            p for p in root.glob("checkpoint-*")
            if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)
        ),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
        reverse=True,
    )

    final_path = root / "final_weights"
    if requested is None:
        if checkpoints or final_path.exists():
            raise RuntimeError(
                "当前 SLAKE Stage 2 输出目录已有 checkpoint/final_weights。"
                "如需恢复请用 --resume_from_checkpoint auto，"
                "如需重训请更换 --run_tag；不要覆盖原结果。"
            )
        return None

    if requested != "auto":
        candidate = Path(requested).resolve()
        if candidate.parent != root or candidate not in checkpoints:
            raise ValueError(
                "只能恢复当前 Stage 2 输出目录内的 checkpoint-N。"
            )
        checkpoints = [candidate]

    for checkpoint in checkpoints:
        manifest_path = checkpoint / "stage2_manifest.json"
        if not manifest_path.is_file():
            continue
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if (
            manifest.get("format_version") != "CM-DA-SSM-STAGE2-SLAKE-V1"
            or manifest.get("architecture_fingerprint")
            != cfg.architecture_fingerprint
            or manifest.get("ablation_id") != cfg.ablation_id
            or manifest.get("seed") != cfg.seed
            or manifest.get("stage1_weights_dir") != cfg.stage1_weights_dir
        ):
            continue

        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        if any(not (checkpoint / item).is_file() for item in required):
            continue

        if not list(checkpoint.glob("rng_state*.pth")):
            continue

        if cfg.enable_visual_adapter:
            adapter_path = checkpoint / "visual_adapter.pt"
            if not adapter_path.is_file():
                continue
            if manifest.get("visual_adapter_sha256") != _sha256(adapter_path):
                continue

        return str(checkpoint)

    raise RuntimeError(
        "找不到当前 SLAKE Stage 2 的完整且匹配的 checkpoint。"
    )


def print_config(cfg, stage1_manifest: dict) -> None:
    ddp_print("\n" + "=" * 72, print_rank=cfg.print_rank)
    ddp_print("🚀 CM-DA-SSM Stage 2 · SLAKE", print_rank=cfg.print_rank)
    ddp_print(
        f"    Ablation={cfg.ablation_id}, Seed={cfg.seed}, "
        f"Stage1Seed={cfg.stage1_seed}, "
        f"Stage1Step={stage1_manifest.get('global_step')}",
        print_rank=cfg.print_rank,
    )
    ddp_print(
        f"    Fusion={cfg.fusion_mode}, Lambda={cfg.lambda_mode}, "
        f"LambdaInit={cfg.lambda_init}, Eta={cfg.eta}, "
        f"RMS={cfg.use_rms_norm}, BioMedCLIP={cfg.requires_biomedclip}",
        print_rank=cfg.print_rank,
    )
    ddp_print(
        f"    Stage1 weights={cfg.stage1_weights_dir}\n"
        f"    Stage2 output={cfg.output_dir}\n"
        f"    Stage2 LR={cfg.lr_tag}",
        print_rank=cfg.print_rank,
    )
    ddp_print("=" * 72, print_rank=cfg.print_rank)


def main():
    options = parse_options()
    cfg = Stage2TrainConfig()
    set_seed(cfg.seed)

    # 无论是否进入 GPU 训练，都先验证 Stage 1 权重、架构和 SHA256。
    stage1_manifest = validate_stage1_for_slake(cfg)
    print_config(cfg, stage1_manifest)

    if options.check_config_only:
        ddp_print("✅ Stage 2 配置 / Stage 1 完整性预检通过。")
        return

    train_dataset = load_train_data(cfg)
    if options.check_data_only:
        ddp_print("✅ SLAKE 训练集独立预检通过。")
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    # 冒烟测试的目录与正式训练目录严格隔离。
    output_dir = (
        cfg.output_dir + f"_smoke_steps{options.smoke_steps}"
        if options.smoke_steps else cfg.output_dir
    )
    resume_dir = (
        None if options.smoke_steps
        else resolve_resume_checkpoint(
            cfg=cfg, output_dir=output_dir,
            requested=options.resume_from_checkpoint,
        )
    )

    # 所有 Rank 使用相同的目录；只在空目录创建之后写入文件。
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from utils.qwen3vl.qwen3_vl_8B_quant_loader import (
        Qwen3VLQuantizedLoader,
    )
    from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_lora_wrapper import (
        Qwen3VLLoraAndCMDASSMWrapper,
    )
    from utils.data_tools.collator.slake.slake_datasets_train_collator import (
        SLAKETrainCollator,
    )
    from transformers import Trainer

    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map={"": local_rank} if local_rank >= 0 else "auto",
    )
    base_model, processor = loader.load()
    processor.tokenizer.padding_side = "right"

    biomed_extractor = biomed_transform = biomed_tokenizer = None
    if cfg.requires_biomedclip:
        from utils.biomedclip.biomed_clip_loader import load_biomedclip

        biomed_extractor, biomed_transform, biomed_tokenizer = load_biomedclip(
            biomedclip_path=cfg.biomedclip_path,
            print_rank=cfg.print_rank,
        )

    wrapper = Qwen3VLLoraAndCMDASSMWrapper(
        **cfg.wrapper_kwargs(biomed_extractor=biomed_extractor)
    )
    model = wrapper.wrap(base_model)

    # 必须完整加载同一组 Stage 1 的 LoRA 和视觉 Adapter。
    stage1_dir = Path(cfg.stage1_weights_dir)
    _load_lora_strict(model, stage1_dir)
    restore_visual_adapter_strict(model, cfg=cfg, directory=stage1_dir)

    # Stage 2 断点恢复时，仅视觉 Adapter 必须手动恢复，
    # LoRA / optimizer / scheduler / RNG 由 Trainer 统一恢复。
    if resume_dir is not None:
        restore_visual_adapter_strict(model, cfg=cfg, directory=resume_dir)

    ddp_print(
        f"✅ 完整继承 Stage 1：LoRA + "
        f"{'Visual Adapter' if cfg.enable_visual_adapter else 'LoRA-only'}",
        print_rank=cfg.print_rank,
    )

    collator = SLAKETrainCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_transform,
        biomed_tokenizer=biomed_tokenizer,
    )

    args = build_training_args(
        cfg=cfg, output_dir=output_dir, smoke_steps=options.smoke_steps,
    )
    optimizer = build_discriminative_adamw(
        model,
        **cfg.optimizer_kwargs(args),
    )
    ddp_print(
        format_optimizer_groups(optimizer),
        print_rank=cfg.print_rank,
    )

    callbacks = (
        [] if options.smoke_steps else [
            create_checkpoint_callback(
                cfg=cfg,
                stage1_manifest=stage1_manifest,
                train_count=len(train_dataset),
            )
        ]
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )

    try:
        trainer.train(resume_from_checkpoint=resume_dir)

        if options.smoke_steps:
            ddp_print(
                f"✅ SLAKE 冒烟测试通过：{options.smoke_steps} 个优化器步。"
                "没有写入正式 final_weights。",
                print_rank=cfg.print_rank,
            )
            return

        # 在所有 rank 中调用 Trainer.save_model，以兼容 DDP/Accelerate。
        final_path = Path(output_dir) / "final_weights"
        trainer.save_model(str(final_path))

        if trainer.is_world_process_zero():
            processor.save_pretrained(str(final_path))
            save_stage2_sidecars(
                model=model,
                directory=final_path,
                cfg=cfg,
                stage1_manifest=stage1_manifest,
                global_step=int(trainer.state.global_step),
                train_count=len(train_dataset),
            )
            _write_json_atomic(
                final_path / "stage2_full_config.json",
                asdict(cfg),
            )
            ddp_print(
                f"🎉 SLAKE Stage 2 完成："
                f"{trainer.state.global_step} steps。"
                f"权重保存至：{final_path}",
                print_rank=cfg.print_rank,
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
