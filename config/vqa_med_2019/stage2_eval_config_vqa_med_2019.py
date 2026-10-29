import os

#解决网络不通
os.environ["http_proxy"]="http://10.110.248.29:7897"
os.environ["https_proxy"]="http://10.110.248.29:7897"

from dataclasses import dataclass

from config.vqa_med_2019.stage2_train_config_vqa_med_2019 import (
    Stage2TrainConfig,
)


@dataclass
class Stage2EvalConfig(Stage2TrainConfig):
    """
    VQA-Med 2019 Stage 2 checkpoint 评测配置。

    直接继承 Stage2TrainConfig，确保训练与评测使用完全一致的：

    - A0～A6 消融定义；
    - LoRA 参数；
    - Conv2D F3/F5/F7 视觉专家；
    - Router、Gate、Lambda 和 RMS 配置；
    - DLR 配置；
    - Stage 2 实验目录命名。
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

    # A6 + 两个 DLR 开关为 True，对应 A6-DLR。
    ablation_id: str = "A6" #"A6" #"A0" #"A6"

    # Stage 2 训练随机种子。
    seed: int = 4096 #2048 #1024 #2048

    # Stage 1 来源权重的随机种子。
    stage1_seed: int = 4096 #2048 #1024 #2048

    # 当前评测 A6-DLR。
    # 若评测普通 A6，将这两个字段改为 False。
    use_discriminative_lr: bool = True
    stage1_use_discriminative_lr: bool = True

    # =========================================================
    # 2. 评测输出
    # =========================================================

    eval_output_root: str = (
        "/home/yuqing/Models/MoRA_Med/"
        "Eval_VQA_MED_2019"
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
        先通过 Stage2TrainConfig 生成对应的 Stage 2 训练目录，
        再将 output_dir 切换为评测结果目录。
        """

        # 根据 A0～A6、DLR、Stage 1 seed 等配置，
        # 生成与训练时完全一致的 Stage 2 实验目录。
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
            "VQA-Med 2019 Stage 2 评测配置"
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
