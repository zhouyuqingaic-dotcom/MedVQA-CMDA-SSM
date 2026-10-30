"""CM-DA-SSM · SLAKE Stage 2 Validation/Test 配置。

继承真实的五组消融 Stage2TrainConfig，因此：
  * 不再使用旧 MoRA-Med 的 A0～A6、Gate、Scale 等参数；
  * 评测严格使用该组 Stage 2 的权重目录和 Stage 1 来源；
  * Validation 选择最佳 checkpoint，Test 可以只读取这一节点。

运行参数：
  --ablation_id A3 --seed 2048 --stage1_seed 2048
  --run_tag main --stage1_run_tag main
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from config.slake.stage2_train_config_slake import Stage2TrainConfig


@dataclass
class Stage2EvalConfig(Stage2TrainConfig):
    """与 Stage 2 正式训练共享所有架构、路径和消融参数。"""

    # Test 默认只读取 Validation 选择的最佳节点；禁止用 Test 选最佳。
    test_checkpoint_mode: str = "best_validation"

    # False：Validation 评测全部完整 checkpoint-* 和 final_weights；
    # True：仅评测最终权重。若已有多个 checkpoint，建议 False。
    eval_final_weights_only: bool = False

    # 不覆盖训练 seed / stage1_seed / ablation_id，继承训练配置的 CLI。
    eval_output_root: str = "/home/yuqing/Models/CM_DA_SSM/Eval_SLAKE"

    # 保持原 MoRA-Med 的 LLM Judge 评测协议。
    # strict_only=True 可进行纯本地字符串匹配冒烟测试，
    # 但它与 LLM Judge 的正式 Overall 指标不可直接比较。
    strict_only: bool = False
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0
    per_device_eval_batch_size: int = 4
    dataloader_num_workers: int = 4

    stage2_run_dir: str = field(init=False)
    eval_run_dir: str = field(init=False)
    validation_output_dir: str = field(init=False)
    test_output_dir: str = field(init=False)
    best_checkpoint_path: str = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.test_checkpoint_mode not in {"best_validation", "all"}:
            raise ValueError(
                "test_checkpoint_mode 只能为 'best_validation' 或 'all'。"
            )

        self.stage2_run_dir = self.output_dir

        # 不能只取最后的 run_main，因为不同架构 / LR / seed
        # 都可能有 run_main，评测结果会相互覆盖。
        relative_run_dir = os.path.relpath(
            self.stage2_run_dir, self.output_root,
        )
        if relative_run_dir.startswith("..") or os.path.isabs(relative_run_dir):
            raise ValueError("Stage 2 训练目录不能位于 output_root 外。")

        self.eval_run_dir = os.path.join(
            self.eval_output_root,
            relative_run_dir,
        )
        self.validation_output_dir = os.path.join(
            self.eval_run_dir, "validation",
        )
        self.test_output_dir = os.path.join(
            self.eval_run_dir, "test",
        )
        self.best_checkpoint_path = os.path.join(
            self.validation_output_dir, "best_checkpoint.json",
        )

        # Eval 不应覆盖继承来的 self.output_dir：仍表示 Stage 2 训练目录。
