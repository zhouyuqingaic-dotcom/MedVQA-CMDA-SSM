"""CM-DA-SSM Stage 2 · SLAKE Validation。

本文件保留原项目的评测入口与核心协议：
  1. Validation 对各 checkpoint 执行医学 VQA 生成；
  2. CLOSED：答案规范化后的严格匹配；
  3. OPEN：严格匹配失败时使用原 DeepSeek LLM Judge；
  4. 输出逐样本 JSONL/CSV、summary、leaderboard；
  5. 在 Validation 上选定最佳 checkpoint，供 Test 使用。

与 MoRA-Med 不同：
  - 只使用 Qwen3VLLoraAndCMDASSMWrapper，加载新的 LoRA + visual_adapter.pt；
  - BioMedCLIP 仅在 A2/A3/A3-NoRMS 使用；A1 是 Uniform Fusion；
  - 记录四方向 RF/RR/CF/CR、全局 λ、逐图 ρ 和注入幅度；
  - pooled + 三条 DeepStack 共享同一 CM-DA-SSM Adapter，按流记录诊断。

诊断的捕捉通过 eval 期间临时包装 Adapter.forward(return_details=True)
实现；它不修改模型参数或 forward 的输出张量。正式训练无需该包装。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.slake.stage2_eval_config_slake import Stage2EvalConfig
from datas.slake_datasets import SLAKEDataset
from utils.data_tools.collator.slake.slake_datasets_eval_collator import (
    SLAKEEvalCollator,
)
from utils.data_tools.prompt_cleaning.slake_answer_cleaning import (
    slake_answer_eval_cleaning,
)
from utils.qwen3vl.qwen3_vl_8B_four_directional_ssm_utils import (
    DIRECTION_NAMES,
)
from utils.training.stage1_checkpoint_utils import (
    get_visual_adapter,
    restore_visual_adapter_strict,
)


DIAGNOSTIC_DIRECTIONS = tuple(DIRECTION_NAMES)
if len(DIAGNOSTIC_DIRECTIONS) != 4:
    raise RuntimeError("CM-DA-SSM 的四方向名称不正确。")


def read_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check_config_only", action="store_true")
    parser.add_argument(
        "--strict_only", action="store_true",
        help="仅用于本地评测/调试；不调用 DeepSeek，不能和 LLM-Judge 正式指标直接比较。",
    )
    options, _ = parser.parse_known_args()
    return options


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def validate_stage2_checkpoint(path: str | Path, cfg: Stage2EvalConfig) -> dict:
    """验证权重来源、架构与 Visual Adapter 哈希；拒绝旧 MoRA-Med 目录。"""
    p = Path(path).resolve()
    root = Path(cfg.stage2_run_dir).resolve()
    if p.parent != root or not (
        p.name == "final_weights"
        or re.fullmatch(r"checkpoint-\d+", p.name)
    ):
        raise RuntimeError(f"权重目录不属于当前 SLAKE Stage 2 运行：{p}")

    manifest_path = p / "stage2_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"缺少 CM-DA-SSM Stage 2 manifest：{manifest_path}。"
            "不要使用旧 MoRA-Med checkpoint。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "format_version": "CM-DA-SSM-STAGE2-SLAKE-V1",
        "dataset": "SLAKE",
        "ablation_id": cfg.ablation_id,
        "architecture_fingerprint": cfg.architecture_fingerprint,
        "stage1_weights_dir": cfg.stage1_weights_dir,
        "seed": cfg.seed,
        "stage1_seed": cfg.stage1_seed,
        "run_tag": cfg.run_tag,
        "stage1_run_tag": cfg.stage1_run_tag,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Stage 2 checkpoint 字段 {key} 不匹配："
                f"expected={value!r} actual={manifest.get(key)!r}"
            )
    if manifest.get("architecture_parameters") != cfg.architecture_parameters():
        raise RuntimeError("Stage 2 checkpoint 完整架构配置不匹配。")

    if not (p / "adapter_config.json").is_file():
        raise FileNotFoundError(f"缺少 LoRA adapter_config.json：{p}")
    lora_files = [
        file for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (file := p / name).is_file()
    ]
    if len(lora_files) != 1:
        raise RuntimeError(f"需要唯一的 Stage 2 LoRA 权重：{p}")

    visual_path = p / "visual_adapter.pt"
    if cfg.enable_visual_adapter:
        if not visual_path.is_file():
            raise FileNotFoundError(f"缺少 Stage 2 Visual Adapter：{visual_path}")
        if manifest.get("visual_adapter_sha256") != _file_sha256(visual_path):
            raise RuntimeError(f"Stage 2 Visual Adapter SHA256 不匹配：{p}")
    elif visual_path.exists():
        raise RuntimeError("A0 checkpoint 不应包含 Visual Adapter。")
    return manifest


def discover_checkpoints(cfg: Stage2EvalConfig) -> list[Path]:
    """Validation 中候选列表固定，仅使用本次 Stage 2 实验目录。"""
    root = Path(cfg.stage2_run_dir)
    if cfg.eval_final_weights_only:
        names = [root / "final_weights"]
    else:
        names = sorted(
            (
                item for item in root.glob("checkpoint-*")
                if item.is_dir() and re.fullmatch(r"checkpoint-\d+", item.name)
            ),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        final_path = root / "final_weights"
        if final_path.is_dir():
            names.append(final_path)
    if not names:
        raise FileNotFoundError(f"没有发现 SLAKE Stage 2 checkpoint：{root}")
    for name in names:
        validate_stage2_checkpoint(name, cfg)
    return names


def _install_generation_forward(model) -> None:
    """generate() 中使用预先缓存的 BioMedCLIP，不重复计算，也不清空缓存。"""
    if not hasattr(model, "original_forward"):
        raise RuntimeError("CM-RADF Wrapper 未安装 original_forward patch。")

    def forward(self, *args, **kwargs):
        kwargs.pop("biomed_image_tensors", None)
        kwargs.pop("biomed_text_tokens", None)
        return self.original_forward(*args, **kwargs)

    model.forward = types.MethodType(forward, model)


def _enable_adapter_diagnostics(adapter):
    """记录同一个 Adapter 的连续四条流：pooled，DeepStack 0/1/2。

    每次视觉前向会复用 adapter.forward 四次。绝不能用最后一次
    DeepStack 诊断覆盖 pooled；按 Adapter 调用顺序逐流追加。
    """
    original = adapter.forward
    holder: dict[str, list] = {"streams": []}

    def wrapped(self, *args, **kwargs):
        kwargs.pop("return_details", None)
        enhanced, details = original(*args, return_details=True, **kwargs)
        holder["streams"].append([
            {
                "routing_weights": value["routing_weights"].detach().float().cpu(),
                "rho": value["rho"].detach().float().cpu(),
                "lambda": value["lambda"].detach().float().cpu(),
                "delta_rms": value["delta_rms"].detach().float().cpu(),
            }
            for value in details
        ])
        return enhanced

    adapter.forward = types.MethodType(wrapped, adapter)
    return holder, original


def _stats_for_image(detail: dict) -> dict:
    """按 token 平均四方向权重，单张图像分别记录 λ、ρ 和残差幅度。"""
    beta = detail["routing_weights"]
    if beta.ndim != 3 or beta.size(0) != 1 or beta.size(-1) != 4:
        raise RuntimeError(f"四方向 routing_weights 形状错误：{tuple(beta.shape)}")
    mean = beta.mean(dim=(0, 1))
    entropy = -(
        beta.clamp_min(1e-12) * beta.clamp_min(1e-12).log()
    ).sum(dim=-1).mean()
    result = {
        f"direction_{name.lower()}": float(mean[i].item())
        for i, name in enumerate(DIAGNOSTIC_DIRECTIONS)
    }
    lam = float(detail["lambda"].item())
    rho = float(detail["rho"].item())
    result.update({
        "lambda_value": lam,
        "rho": rho,
        "lambda_times_rho": lam * rho,
        "injected_residual_rms": float(detail["delta_rms"].item()),
        "routing_entropy": float(entropy.item()),
        "routing_entropy_normalized": float(entropy.item() / math.log(4)),
    })
    return result


def _diagnostic_field_names(prefix: str = "") -> list[str]:
    """生成 pooled/DeepStack 一致的逐样本诊断字段。"""
    names = [
        *(f"direction_{d.lower()}" for d in DIAGNOSTIC_DIRECTIONS),
        "lambda_value", "rho", "lambda_times_rho",
        "injected_residual_rms", "routing_entropy",
        "routing_entropy_normalized",
    ]
    return [f"{prefix}_{name}" if prefix else name for name in names]


def build_model_for_checkpoint(cfg: Stage2EvalConfig, checkpoint: Path):
    """重建干净 Qwen3-VL/PEFT，并严格加载 Stage 2 当前 checkpoint 的权重。"""
    from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
    from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_lora_wrapper import (
        Qwen3VLLoraAndCMDASSMWrapper,
    )
    from training.stage2_trainer_slake import _load_lora_strict

    validate_stage2_checkpoint(checkpoint, cfg)

    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map="auto",
    )
    base_model, processor = loader.load()
    processor.tokenizer.padding_side = "left"

    biomed_extractor = biomed_transform = biomed_tokenizer = None
    if cfg.requires_biomedclip:
        from utils.biomedclip.biomed_clip_loader import load_biomedclip
        biomed_extractor, biomed_transform, biomed_tokenizer = load_biomedclip(
            biomedclip_path=cfg.biomedclip_path,
            print_rank=cfg.print_rank,
        )

    kwargs = cfg.wrapper_kwargs(biomed_extractor=biomed_extractor)
    # 评测只做 inference，不需要 gradient checkpointing。
    kwargs["gradient_checkpointing"] = False
    model = Qwen3VLLoraAndCMDASSMWrapper(**kwargs).wrap(base_model)

    _load_lora_strict(model, checkpoint)
    restore_visual_adapter_strict(model, cfg=cfg, directory=checkpoint)
    model.requires_grad_(False)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    if cfg.requires_biomedclip:
        _install_generation_forward(model)

    return model, processor, biomed_transform, biomed_tokenizer


def _create_loader(cfg: Stage2EvalConfig, processor, transform, tokenizer, split: str):
    path = (
        cfg.slake_val_json_path if split == "validation"
        else cfg.slake_test_json_path
    )
    dataset = SLAKEDataset(json_path=path, image_root=cfg.slake_image_root)
    if len(dataset) == 0:
        raise RuntimeError(f"SLAKE {split} 数据集为空。")
    collator = SLAKEEvalCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=transform,
        biomed_tokenizer=tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.per_device_eval_batch_size,
        collate_fn=collator,
        num_workers=cfg.dataloader_num_workers,
        shuffle=False,
    )


def get_llm_judge(strict_only: bool):
    """保留原 MoRA-Med 的 DeepSeek Judge 协议；离线排查可禁用。"""
    if strict_only:
        return None, None
    from config.LLM_config import LLMAPIConfig
    from LLM_api.deepseek import DeepSeekClient
    cfg = LLMAPIConfig()
    return DeepSeekClient.from_config(cfg), cfg


def judge_records(records: list[dict], llm_client, llm_config) -> None:
    """与原脚本相同：OPEN 且严格匹配失败时调用 LLM 判定。"""
    if llm_client is None:
        return
    from LLM_api.prompts.slake_prompt_builder_deepseek import (
        build_llm_judge_user_prompt,
        parse_llm_judge_response,
    )

    for row in tqdm(records, desc="SLAKE LLM Judge"):
        if row["question_category"] != "open" or row["is_norm_match"]:
            continue
        prompt = build_llm_judge_user_prompt(
            question=row["question"],
            gt_raw=row["gt_raw"],
            gt_norm=row["gt_norm"],
            pred_raw=row["pred_raw"],
            pred_norm=row["pred_norm"],
        )
        response = llm_client.ask(
            llm_config.medical_vqa_llm_judge_system_prompt, prompt,
        )
        parsed = parse_llm_judge_response(response)
        row["llm_judge_raw_response"] = response
        row["llm_judge_score"] = parsed.get("score")
        row["llm_judge_reason"] = (
            parsed.get("reasoning") or parsed.get("reason")
            or parsed.get("explanation")
        )
        row["final_correct"] = parsed.get("score") == "correct"


def calculate_metrics(records: list[dict], cfg: Stage2EvalConfig) -> dict:
    if not records:
        raise RuntimeError("没有可评估的 SLAKE 样本。")
    closed = [row for row in records if row["question_category"] == "closed"]
    opened = [row for row in records if row["question_category"] == "open"]
    correct = lambda rows: sum(bool(row["final_correct"]) for row in rows)
    metrics = {
        "metric_protocol": (
            "strict_only" if cfg.strict_only else "strict_plus_deepseek_judge"
        ),
        "closed_count": len(closed),
        "open_count": len(opened),
        "closed_acc": correct(closed) / len(closed) if closed else None,
        "open_strict_acc": correct(opened) / len(opened) if opened else None,
        "overall_strict_acc": correct(records) / len(records),
    }
    diagnostics = [row for row in records if row.get("lambda_value") is not None]
    if diagnostics:
        metrics["routing"] = {
            f"mean_{name.lower()}": sum(
                float(row[f"direction_{name.lower()}"]) for row in diagnostics
            ) / len(diagnostics)
            for name in DIAGNOSTIC_DIRECTIONS
        }
        for metric in (
            "lambda_value", "rho", "lambda_times_rho",
            "injected_residual_rms", "routing_entropy_normalized",
        ):
            metrics["routing"][f"mean_{metric}"] = sum(
                float(row[metric]) for row in diagnostics
            ) / len(diagnostics)
    # 新增多层诊断不改变既有 routing 字段：其含义固定为 pooled。
    # DeepStack 0/1/2 的指标单独归档，避免被误当成 pooled 注入统计。
    by_stream = {}
    for stream_index in range(3):
        prefix = f"deepstack_{stream_index}"
        selected = [
            row for row in records
            if row.get(f"{prefix}_lambda_value") is not None
        ]
        if not selected:
            continue
        by_stream[prefix] = {
            f"mean_{name}": sum(float(row[f"{prefix}_{name}"]) for row in selected)
            / len(selected)
            for name in _diagnostic_field_names()
        }
    if by_stream:
        metrics["routing_by_stream"] = by_stream
    return metrics


def write_results(folder: Path, records: list[dict], summary: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "samples.jsonl").open("w", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (folder / "samples.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate_checkpoint(
    checkpoint: Path,
    cfg: Stage2EvalConfig,
    *,
    split: str,
    llm_client,
    llm_cfg,
) -> dict:
    """对 Validation 或 Test 单个 checkpoint 执行完整生成与评测。"""
    if split not in {"validation", "test"}:
        raise ValueError(f"未知评测 split：{split}")
    validate_stage2_checkpoint(checkpoint, cfg)
    output_root = Path(
        cfg.validation_output_dir if split == "validation"
        else cfg.test_output_dir
    )
    output = output_root / checkpoint.name
    print(f"\n🔎 SLAKE {split} · {checkpoint.name}")
    model, processor, transform, tokenizer = build_model_for_checkpoint(
        cfg, checkpoint
    )
    loader = _create_loader(cfg, processor, transform, tokenizer, split)
    adapter = get_visual_adapter(model)
    vision = model.base_model.model.model.visual
    device = next(vision.parameters()).device

    holder = None
    original_adapter_forward = None
    if adapter is not None:
        holder, original_adapter_forward = _enable_adapter_diagnostics(adapter)

    records: list[dict] = []
    try:
        with torch.inference_mode():
            for inputs, metadata in tqdm(loader, desc=f"SLAKE {split}"):
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
                if holder is not None:
                    holder["streams"] = []
                # 每个生成 batch 都必须清空图文条件，防止前一个 batch 泄漏。
                vision.current_biomed_img_feat = None
                vision.current_biomed_txt_feat = None

                try:
                    if cfg.requires_biomedclip:
                        image = inputs.pop("biomed_image_tensors")
                        text = inputs.pop("biomed_text_tokens")
                        biomed_model = model.biomed_extractor
                        reference = next(biomed_model.parameters())
                        image = image.to(reference.device, dtype=reference.dtype)
                        text = text.to(reference.device)
                        image_feat, text_feat = biomed_model(image, text)
                        vision.current_biomed_img_feat = image_feat.detach()
                        vision.current_biomed_txt_feat = text_feat.detach()

                    generate_kwargs: dict[str, Any] = {
                        "max_new_tokens": cfg.max_new_tokens,
                        "do_sample": cfg.do_sample,
                    }
                    if cfg.do_sample:
                        generate_kwargs["temperature"] = cfg.temperature

                    generated = model.generate(**inputs, **generate_kwargs)
                    new_tokens = [
                        output_ids[len(input_ids):]
                        for input_ids, output_ids in zip(inputs["input_ids"], generated)
                    ]
                    predictions = processor.batch_decode(
                        new_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if len(predictions) != len(metadata):
                        raise RuntimeError("SLAKE 解码数量与样本数量不一致。")

                    # 一条 SLAKE 样本对应一张图；每条流都应具有
                    # 与当前 batch 一一对齐的单图诊断。
                    streams = holder["streams"] if holder is not None else []
                    count = len(getattr(vision, "deepstack_visual_indexes", (0, 1, 2)))
                    stream_names = ("pooled",) + tuple(
                        f"deepstack_{i}" for i in range(count)
                    )
                    if adapter is not None:
                        if len(streams) != len(stream_names):
                            raise RuntimeError(
                                "共享 Adapter 诊断流数量不匹配："
                                f"actual={len(streams)}, expected={len(stream_names)} "
                                f"({stream_names})"
                            )
                        if any(len(details) != len(metadata) for details in streams):
                            raise RuntimeError(
                                "多流四方向诊断与 SLAKE 样本数不一致；"
                                f"stream_counts={[len(d) for d in streams]}, "
                                f"batch={len(metadata)}"
                            )

                    for index, (prediction, meta) in enumerate(
                        zip(predictions, metadata)
                    ):
                        gt_raw = str(meta.get("gt_answer", meta.get("answer", "")))
                        gt_norm = slake_answer_eval_cleaning(gt_raw)
                        pred_norm = slake_answer_eval_cleaning(prediction)
                        matched = gt_norm == pred_norm
                        category = (
                            "closed" if str(meta.get("answer_type", ""))
                            .strip().upper() == "CLOSED" else "open"
                        )
                        row = {
                            "index": meta.get("index"),
                            "image_path": meta.get("image_path"),
                            "question": meta["question"],
                            "question_category": category,
                            "gt_raw": gt_raw,
                            "gt_norm": gt_norm,
                            "pred_raw": prediction.strip(),
                            "pred_norm": pred_norm,
                            "is_norm_match": matched,
                            "llm_judge_score": None,
                            "llm_judge_reason": None,
                            "llm_judge_raw_response": None,
                            "final_correct": matched,
                        }
                        # pooled 使用原有列名；其余三条流单独加前缀。
                        # 保持每个样本 CSV 列集合相同。
                        diagnostic_keys = _diagnostic_field_names()
                        for name in stream_names[1:]:
                            diagnostic_keys.extend(_diagnostic_field_names(name))
                        row.update({key: None for key in diagnostic_keys})
                        if adapter is not None:
                            row.update(_stats_for_image(streams[0][index]))
                            for stream_index, name in enumerate(stream_names[1:], start=1):
                                for key, val in _stats_for_image(
                                    streams[stream_index][index]
                                ).items():
                                    row[f"{name}_{key}"] = val
                        records.append(row)
                finally:
                    vision.current_biomed_img_feat = None
                    vision.current_biomed_txt_feat = None
    finally:
        if adapter is not None:
            adapter.forward = original_adapter_forward
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    judge_records(records, llm_client=llm_client, llm_config=llm_cfg)
    result = {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint.resolve()),
        "split": split,
        "ablation_id": cfg.ablation_id,
        "seed": cfg.seed,
        "stage1_seed": cfg.stage1_seed,
        **calculate_metrics(records, cfg),
    }
    write_results(output, records, result)
    print(
        f"✅ {checkpoint.name} · Overall: {result['overall_strict_acc']:.4%} "
        f"· Protocol: {result['metric_protocol']}"
    )
    return result


def main():
    options = read_options()
    cfg = Stage2EvalConfig(strict_only=options.strict_only)
    checkpoints = discover_checkpoints(cfg)
    print("Validation 候选 checkpoint：", [item.name for item in checkpoints])
    if options.check_config_only:
        print("✅ Validation 配置与所有 checkpoint manifest 通过预检")
        return

    llm_client, llm_cfg = get_llm_judge(cfg.strict_only)
    results = [
        evaluate_checkpoint(
            item, cfg, split="validation",
            llm_client=llm_client, llm_cfg=llm_cfg,
        )
        for item in checkpoints
    ]
    root = Path(cfg.validation_output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "leaderboard.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 仅使用 Validation 指标选最佳，Test 绝不可参与选点。
    best = max(results, key=lambda x: x["overall_strict_acc"])
    selected = {
        "dataset": "SLAKE",
        "selection_split": "validation",
        "selection_metric": "overall_strict_acc",
        "metric_protocol": best["metric_protocol"],
        "selected_checkpoint": best["checkpoint"],
        "selected_checkpoint_path": best["checkpoint_path"],
        "validation_score": best["overall_strict_acc"],
        "ablation_id": cfg.ablation_id,
        "architecture_fingerprint": cfg.architecture_fingerprint,
        "seed": cfg.seed,
        "stage1_seed": cfg.stage1_seed,
        "run_tag": cfg.run_tag,
    }
    Path(cfg.best_checkpoint_path).write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("🎉 Validation 选出的最佳 checkpoint：", best["checkpoint"])
    print("最佳节点记录：", cfg.best_checkpoint_path)


if __name__ == "__main__":
    main()
