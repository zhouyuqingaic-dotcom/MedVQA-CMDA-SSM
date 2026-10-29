"""
CM-DA-SSM Stage 1 — MIMIC-CXR 报告监督配置。

本文件只保留用户新五组消融：A0、A1、A2、A3、A3-NoRMS。
每组从同一个 Qwen3-VL 基座独立训练，不能跨组复用 Stage 1 权重。

数据集原有的 IMPRESSION 目标、缓存及图像路径保留。
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

from config.cm_da_ssm.ablation_config import (
    ABLATION_IDS,
    CMDASSMCommonConfig,
)


def parse_args():
    """兼容原训练入口传入 --ablation_id / --seed 的习惯。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ablation_id", type=str, default="A3", choices=ABLATION_IDS)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--run_tag", type=str, default="main")
    args, _ = parser.parse_known_args()
    return args


_cli_args = parse_args()


@dataclass
class TrainConfig(CMDASSMCommonConfig):
    """
    Stage 1：MIMIC-CXR → IMPRESSION 监督训练。

    BioMedCLIP Text Encoder 的输入为固定的报告生成指令，
    严禁把目标 IMPRESSION 放入 Query，防止目标泄露。
    """

    # =========================================================
    # 1. 消融实验与随机种子
    # =========================================================
    ablation_id: str = _cli_args.ablation_id
    seed: int = _cli_args.seed
    run_tag: str = _cli_args.run_tag

    # =========================================================
    # 2. MIMIC-CXR 路径与处理配置（来自上传项目）
    # =========================================================
    mimic_cxr_root: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0"

    mimic_cxr_metadata_csv: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"

    mimic_cxr_image_root: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/files"

    mimic_cxr_report_root: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/reports"

    mimic_cxr_target_section: str = "impression"

    mimic_cxr_load_report: bool = True

    mimic_cxr_view_positions: Optional[list[str]] = None

    mimic_cxr_drop_empty_target: bool = True

    mimic_cxr_cache_dir: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/cache"

    mimic_cxr_use_indices_cache: bool = True

    mimic_cxr_rebuild_indices_cache: bool = False

    mimic_cxr_cache_prefix: str = "mimic_cxr_train_clean_v2_screen160k_seed2048"

    mimic_cxr_instruction_suffix: str = (
        "Act as an expert radiologist. "
        "Carefully analyze this chest radiograph and provide a comprehensive clinical interpretation."
    )

    mimic_cxr_max_size: int = 1024

    max_mimic_cxr_train_samples: Optional[int] = None

    # =========================================================
    # 3. Stage 1 的优化器 LR recipe（五组 SSM 统一，A0 无 SSM 组）
    # =========================================================
    # 恢复到此前已通过短程训练验证的 SSM 学习率 1e-4。
    # 各组实际学习率由共享配置的 lr_tag 动态写入独立目录层级，
    # 不再在 lr_recipe_name 里硬编码单个学习率。
    # Stage 2 的 SSM 学习率仍然是 3e-5。
    lr_recipe_name: str = "CMDA-DLR"
    ssm_learning_rate: float = 1e-4

    # =========================================================
    # 4. Stage 1 的共同训练预算（五组相同）
    # =========================================================
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 1.0
    max_steps: int = -1

    warmup_steps: int = 200
    save_steps: int = 500
    save_total_limit: int = 2
    dataloader_num_workers: int = 12

    # 固定报告指令；只有 A2/A3/A3-NoRMS 需要实际编码。
    biomed_condition_text_mode: str = "fixed_instruction"

    output_dir: str = field(init=False)
    final_weights_dir: str = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.mimic_cxr_target_section.lower() != "impression":
            raise ValueError("Stage 1 必须监督 IMPRESSION。")

        if self.biomed_condition_text_mode != "fixed_instruction":
            raise ValueError(
                "Stage 1 的 BiomedCLIP 文本只能是固定报告指令，"
                "不能使用真实目标 IMPRESSION。"
            )

        if (
            self.max_mimic_cxr_train_samples is not None
            and self.max_mimic_cxr_train_samples <= 0
        ):
            raise ValueError("max_mimic_cxr_train_samples 必须为正数或 None。")

        # 五组每组有独立目录，包含结构指纹、真实分组 LR、seed、run_tag。
        self.output_dir = self.stage1_run_dir(
            output_root=self.output_root,
            cache_prefix=self.mimic_cxr_cache_prefix,
            architecture_tag=self.architecture_tag,
            seed=self.seed,
            lr_tag=self.lr_tag,
            run_tag=self.run_tag,
        )

        self.final_weights_dir = os.path.join(
            self.output_dir,
            "final_weights",
        )