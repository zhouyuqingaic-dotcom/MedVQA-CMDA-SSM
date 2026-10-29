import os
from dataclasses import dataclass

from config.slake.stage2_train_config_slake import Stage2TrainConfig


@dataclass
class Stage2EvalConfig(Stage2TrainConfig):
    """
    SLAKE Stage 2 checkpoint 评测配置。

    直接继承 Stage2TrainConfig，确保：
    - A0～A5 消融定义一致；
    - LoRA 结构一致；
    - RoMA-Net V2-lite 结构一致；
    - Stage 2 权重目录命名一致。
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

    # 必须对应要评测的 Stage 2 实验
    ablation_id: str = "A0" #"A1" #"A2" #"A3" #"A4" #"A5" #"A6" #"A0" #"A0" #"A5"

    # 当前评测 A6
    # use_discriminative_lr: bool = True
    # stage1_use_discriminative_lr: bool = True
    #为A0,A1择为False
    use_discriminative_lr: bool = False
    stage1_use_discriminative_lr: bool = False

    # Stage 2 训练种子
    seed: int = 1024 #4096 #1024 #4096 #1024 #2048

    # Stage 1 来源权重的种子
    stage1_seed: int = 1024 #4096 #1024 #4096 #1024 #2048

    # True: 只评测 final_weights
    # False: 评测 checkpoint-* 和 final_weights
    eval_final_weights_only: bool = False

    # 评测结果根目录
    eval_output_root: str = (
        "/home/yuqing/Models/MoRA_Med/Eval_SLAKE"
    )

    # 生成参数
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    # DataLoader
    per_device_eval_batch_size: int = 4
    dataloader_num_workers: int = 4

    def __post_init__(self):
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
            "SLAKE Stage 2 评测配置"
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