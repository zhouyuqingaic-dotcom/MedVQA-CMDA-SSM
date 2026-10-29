"""
列出 CM-DA-SSM 五组消融的两阶段训练矩阵。

默认一个随机种子：
    5 组 × 1 个 MIMIC-CXR Stage 1 = 5 次 Stage 1
    5 组 × 4 个 Med-VQA 数据集 = 20 次 Stage 2

合计 25 个独立训练任务，不等于 25 GPU 小时。

使用示例：
    python examples/print_five_ablation_plan.py
    python examples/print_five_ablation_plan.py --seeds 2048 4096
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.cm_da_ssm.ablation_config import ABLATION_IDS
from config.stage1_train_config_mimic_cxr import TrainConfig
from config.slake.stage2_train_config_slake import Stage2TrainConfig as SLAKE
from config.vqa_rad.stage2_train_config_vqa_rad import Stage2TrainConfig as RAD
from config.vqa_med_2019.stage2_train_config_vqa_med_2019 import (
    Stage2TrainConfig as MED2019,
)
from config.vqa_med_2021.stage2_train_config_vqa_med_2021 import (
    Stage2TrainConfig as MED2021,
)


DATASETS = (SLAKE, RAD, MED2019, MED2021)


def build_training_plan(seeds=(2048,), run_tag="main"):
    """
    返回任务配置列表，不加载数据集、不创建模型、不启动训练。

    每个 stage2 config 的 stage1_weights_dir 必须严格指向
    同一 seed、同一消融的 stage1.final_weights_dir。
    """
    stage1_jobs = []
    stage2_jobs = []

    for seed in seeds:
        for ablation in ABLATION_IDS:
            cfg1 = TrainConfig(
                ablation_id=ablation,
                seed=seed,
                run_tag=run_tag,
            )
            stage1_jobs.append(cfg1)

            for dataset_cls in DATASETS:
                cfg2 = dataset_cls(
                    ablation_id=ablation,
                    seed=seed,
                    stage1_seed=seed,
                    run_tag=run_tag,
                    stage1_run_tag=run_tag,
                )

                if cfg2.stage1_weights_dir != cfg1.final_weights_dir:
                    raise RuntimeError(
                        "Stage 2 与自身 Stage 1 的权重目录不一致！"
                    )

                stage2_jobs.append(cfg2)

    return stage1_jobs, stage2_jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[2048])
    parser.add_argument("--run-tag", type=str, default="main")
    args = parser.parse_args()

    s1_jobs, s2_jobs = build_training_plan(
        seeds=args.seeds,
        run_tag=args.run_tag,
    )

    print("=" * 85)
    print("CM-DA-SSM Five-Ablation Two-Stage Training Plan")
    print("=" * 85)

    for c in s1_jobs:
        print(
            "[Stage 1]",
            c.ablation_id,
            f"seed={c.seed}",
            f"fusion={c.fusion_mode}",
            f"lambda={c.lambda_mode}",
            f"rms={c.use_rms_norm}",
        )
        print("   output:", c.final_weights_dir)

    for c in s2_jobs:
        print(
            "[Stage 2]",
            c.ablation_id,
            c.dataset_name,
            f"seed={c.seed}",
        )
        print("   inherit:", c.stage1_weights_dir)
        print("   output :", c.output_dir)

    print("=" * 85)
    print("Stage 1 runs:", len(s1_jobs))
    print("Stage 2 runs:", len(s2_jobs))
    print("Total training jobs:", len(s1_jobs) + len(s2_jobs))


if __name__ == "__main__":
    main()
