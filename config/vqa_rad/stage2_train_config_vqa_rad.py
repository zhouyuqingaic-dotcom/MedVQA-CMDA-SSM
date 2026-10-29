"""
CM-DA-SSM Stage 2 — VQA-RAD。

五组消融每组独立继承自己 Stage 1 MIMIC-CXR 检查点；
不允许 A3-NoRMS 读取 A3，也不允许任何组借用其他组权重。

该文件目前是新训练配置；原 Trainer 后续还需同步切换 Wrapper、
Optimizer 和 Stage 1 Checkpoint 加载路径。
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional

from config.cm_da_ssm.ablation_config import (
    ABLATION_IDS,
    CMDASSMStage2BaseConfig,
)


def parse_args():
    """支持旧训练入口的命令行 --ablation_id / --seed / --stage1_seed。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ablation_id", type=str, default="A3", choices=ABLATION_IDS)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--stage1_seed", type=int, default=2048)
    parser.add_argument("--run_tag", type=str, default="main")
    parser.add_argument("--stage1_run_tag", type=str, default="main")
    args, _ = parser.parse_known_args()
    return args


_cli_args = parse_args()


@dataclass
class Stage2TrainConfig(CMDASSMStage2BaseConfig):
    """
    VQA-RAD 的五组 CM-DA-SSM 实验配置。

    Stage 2 用当前问题文本作为 CM-RADF 的 BiomedCLIP Query；
    不能输入标签中的目标答案。A0/A1 无需冻结 BioMedCLIP。
    """

    # =========================================================
    # 1. 实验身份、种子及 Stage 1 来源
    # =========================================================
    dataset_name: str = "vqa_rad"
    ablation_id: str = _cli_args.ablation_id
    seed: int = _cli_args.seed
    stage1_seed: int = _cli_args.stage1_seed
    run_tag: str = _cli_args.run_tag
    stage1_run_tag: str = _cli_args.stage1_run_tag

    # =========================================================
    # 2. 数据集路径与处理设置（来自本次上传的工程）
    # =========================================================
    vqa_rad_train_jsonl_path: str = "/home/yuqing/Datas/VQA-RAD/train.jsonl"

    vqa_rad_test_jsonl_path: str = "/home/yuqing/Datas/VQA-RAD/test_official.jsonl"

    vqa_rad_image_root: str = "/home/yuqing/Datas/VQA-RAD/images"

    vqa_rad_max_size: int = 1024

    vqa_rad_instruction_suffix: str = (
        "Answer the question briefly and directly based on the image. "
        "Use a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no. "
        "Do not add unnecessary explanation."
    )

    max_vqa_rad_train_samples: Optional[int] = None

    # =========================================================
    # 3. Stage 2 共同实验预算：同一个数据集五组完全一致
    # =========================================================
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 10.0
    warmup_steps: int = 50
    save_steps: int = 250
    save_total_limit: int = 40
    dataloader_num_workers: int = 8

    max_steps: int = -1

    # 使用与该组 Stage 1 完全相同的架构，只降低 Stage 2 LR。
    # 学习率差异由公共基类处理，五组统一。
    learning_rate: float = 1e-5
    lora_learning_rate: float = 1e-5
    ssm_learning_rate: float = 3e-5
    visual_projection_learning_rate: float = 3e-5
    cm_radf_learning_rate: float = 2e-5
    lambda_learning_rate: float = 2e-5
    visual_norm_learning_rate: float = 1e-5

    # 全部 Stage 2 只读取该样本的问题，不读取目标答案。
    biomed_condition_text_mode: str = "question"
