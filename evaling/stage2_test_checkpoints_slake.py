"""CM-DA-SSM Stage 2 · SLAKE Test。

与 Validation 使用同一套模型加载 / BioMedCLIP / 4D-SSM 诊断 / 评测协议。
默认 Test 只读取 Validation 的 best_checkpoint.json；不能根据 Test 选点。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.slake.stage2_eval_config_slake import Stage2EvalConfig
from evaling.stage2_validate_checkpoints_slake import (
    discover_checkpoints,
    evaluate_checkpoint,
    get_llm_judge,
    validate_stage2_checkpoint,
)


def read_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check_config_only", action="store_true")
    parser.add_argument(
        "--strict_only", action="store_true",
        help="仅供无外网本地排查，与正式 LLM Judge 的指标不直接可比。",
    )
    parser.add_argument(
        "--test_mode", choices=("best_validation", "all"),
        default="best_validation",
        help="默认只测试在 Validation 上选出的最佳节点。",
    )
    options, _ = parser.parse_known_args()
    return options


def choose_test_checkpoints(cfg: Stage2EvalConfig) -> list[Path]:
    if cfg.test_checkpoint_mode == "all":
        return discover_checkpoints(cfg)

    path = Path(cfg.best_checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            "未找到 Validation 选择结果 best_checkpoint.json；"
            "请先运行 stage2_validate_checkpoints_slake.py。"
            f"路径：{path}"
        )
    best = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "SLAKE",
        "selection_split": "validation",
        "ablation_id": cfg.ablation_id,
        "architecture_fingerprint": cfg.architecture_fingerprint,
        "seed": cfg.seed,
        "stage1_seed": cfg.stage1_seed,
        "run_tag": cfg.run_tag,
    }
    for name, value in expected.items():
        if best.get(name) != value:
            raise RuntimeError(
                f"Validation best checkpoint 元数据不匹配：{name}"
            )

    # 如果 Validation 用 LLM Judge 选点，Test 不应突然按严格匹配
    # 宣称两者指标一致。允许两协议都被记录，但显式提示。
    if best.get("metric_protocol") != (
        "strict_only" if cfg.strict_only else "strict_plus_deepseek_judge"
    ):
        raise RuntimeError(
            "Test 与 Validation 的评测协议不同，"
            "请使用相同的 --strict_only 设置，或重新运行 Validation。"
        )

    candidate = Path(best["selected_checkpoint_path"]).resolve()
    if candidate.name != best.get("selected_checkpoint"):
        raise RuntimeError("best_checkpoint.json 的目录名与路径不一致。")

    validate_stage2_checkpoint(candidate, cfg)
    return [candidate]


def main():
    options = read_options()
    cfg = Stage2EvalConfig(
        strict_only=options.strict_only,
        test_checkpoint_mode=options.test_mode,
    )
    checkpoints = choose_test_checkpoints(cfg)
    print("Test 评测节点：", [p.name for p in checkpoints])
    if options.check_config_only:
        print("✅ Test 配置及 checkpoint 校验通过。")
        return

    client, llm_cfg = get_llm_judge(cfg.strict_only)
    results = [
        evaluate_checkpoint(
            p, cfg, split="test",
            llm_client=client, llm_cfg=llm_cfg,
        )
        for p in checkpoints
    ]

    output = Path(cfg.test_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "leaderboard.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("🎉 SLAKE Test 已完成，结果目录：", output)


if __name__ == "__main__":
    main()
