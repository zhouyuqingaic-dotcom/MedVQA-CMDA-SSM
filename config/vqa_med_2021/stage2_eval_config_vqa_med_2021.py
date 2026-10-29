import os

#解决网络不通
os.environ["http_proxy"]="http://10.110.248.29:7897"
os.environ["https_proxy"]="http://10.110.248.29:7897"

from dataclasses import dataclass

from config.vqa_med_2021.stage2_train_config_vqa_med_2021 import (
    Stage2TrainConfig,
)


@dataclass
class Stage2EvalConfig(Stage2TrainConfig):
    """
    VQA-Med 2021 Task 1 Stage 2 checkpoint 评测配置。

    结构与 SLAKE、VQA-Med 2019 的 EvalConfig 保持一致：

    - 直接继承 Stage2TrainConfig；
    - 使用同一套 A0～A6 消融定义；
    - 使用同一套 LoRA、Visual Adapter、Router、Gate、Lambda、RMS；
    - 根据 Stage2TrainConfig 自动定位 Stage 2 训练目录；
    - Stage2TrainConfig 负责定位训练实验；
    - Validation 脚本评测 checkpoint-* + final_weights，并生成 best_checkpoint.json；
    - Test 脚本根据 test_checkpoint_mode，选择 Validation 最佳节点或全部节点进行测试。

    注意：
    Train 与 Validation 如何组合，只在
    training/stage2_trainer_vqa_med_2021.py 的
    ConcatDataset 列表中设置，与本评测配置无关。
    """
    # =========================================================
    # Test checkpoint 选择方式
    # =========================================================

    # "best_validation":
    #   读取 Validation 评测生成的 best_checkpoint.json，
    #   只测试 Validation 最佳节点。
    #
    # "all":
    #   在 Test 上评测全部 checkpoint-* + final_weights。
    test_checkpoint_mode: str = "best_validation"

    # =========================================================
    # 1. 要评测的 Stage 2 实验
    # =========================================================

    # 常用：
    #   "A0"：LoRA-only baseline
    #   "A6"：MoRA 强视觉残差
    ablation_id: str = "A0" #"A6" #"A0" #"A6" #"A0"  # "A6"

    # Stage 2 训练随机种子。
    seed: int = 4096 #2048 #1024 #2048 #1024 #2048 #4096 #2048 #1024 #2048

    # Stage 1 来源权重随机种子。
    stage1_seed: int = 4096 #2048 #1024 #2048 #1024 #2048 #4096 #2048 #1024 #2048

    # 评测 A6-DLR 时，将以下两个字段改为 True。
    # 普通 A0 / A6 保持 False。
    use_discriminative_lr: bool = False #True
    stage1_use_discriminative_lr: bool = False #True

    # =========================================================
    # 2. 评测输出
    # =========================================================

    eval_output_root: str = (
        "/home/yuqing/Models/MoRA_Med/"
        "Eval_VQA_MED_2021"
    )


    # =========================================================
    # 3. 生成参数
    # =========================================================

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    # =========================================================
    # 4. DataLoader
    # =========================================================

    per_device_eval_batch_size: int = 4
    dataloader_num_workers: int = 4

    def __post_init__(self):
        """
        先由 Stage2TrainConfig 生成对应的 Stage 2 训练目录，
        再将 output_dir 切换为统一评测结果目录。
        """
        super().__post_init__()

        valid_test_checkpoint_modes = {
            "best_validation",
            "all",
        }

        if (
                self.test_checkpoint_mode
                not in valid_test_checkpoint_modes
        ):
            raise ValueError(
                "test_checkpoint_mode 必须是 "
                "'best_validation' 或 'all'，"
                f"当前为 {self.test_checkpoint_mode!r}。"
            )

        self.stage2_run_dir = self.output_dir

        run_name = os.path.basename(
            os.path.normpath(
                self.stage2_run_dir
            )
        )

        # 当前 Stage 2 实验的统一评测根目录。
        self.eval_run_dir = os.path.join(
            self.eval_output_root,
            run_name,
        )

        # Validation 和 Test 分开保存，避免结果互相覆盖。
        self.validation_output_dir = os.path.join(
            self.eval_run_dir,
            "validation",
        )

        self.test_output_dir = os.path.join(
            self.eval_run_dir,
            "test",
        )

        # Validation 脚本生成，Test(best_validation) 模式读取。
        self.best_checkpoint_path = os.path.join(
            self.validation_output_dir,
            "best_checkpoint.json",
        )

        print("\n" + "=" * 60)
        print(
            "VQA-Med 2021 Stage 2 评测配置"
            f" | 消融={self.ablation_id}"
        )
        print(
            f"Stage 2 训练目录："
            f"{self.stage2_run_dir}"
        )
        print(
            f"Validation 输出目录："
            f"{self.validation_output_dir}"
        )
        print(
            f"Test 输出目录："
            f"{self.test_output_dir}"
        )
        print(
            f"Test checkpoint mode："
            f"{self.test_checkpoint_mode}"
        )
        print("=" * 60 + "\n")
