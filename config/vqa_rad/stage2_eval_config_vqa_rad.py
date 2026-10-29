import os
from dataclasses import dataclass, field

from config.vqa_rad.stage2_train_config_vqa_rad import (
    Stage2TrainConfig,
)


@dataclass
class Stage2EvalConfig(Stage2TrainConfig):
    """
    VQA-RAD Stage 2 评测配置。

    直接继承新版 Stage2TrainConfig，确保：

    - A0～A6 消融定义与训练完全一致；
    - Conv2D F3/F5/F7 视觉专家结构一致；
    - LoRA 参数一致；
    - Router、Gate、lambda、RMS 配置一致；
    - DLR 实验命名一致；
    - Stage 1 和 Stage 2 路径生成规则一致。

    本类只负责增加评测专属参数，并将训练输出目录
    转换为权重读取目录和评测结果目录。
    """

    # =========================================================
    # 1. 要评测的实验身份
    # 必须与实际训练完成的 Stage 2 实验一致
    # =========================================================

    # 主实验：A6-DLR
    ablation_id: str = "A0" #"A6" #"A0"

    # Stage 2 训练时使用的 seed
    seed: int = 2048

    # Stage 1 权重对应的 seed
    stage1_seed: int = 2048

    # Stage 1 使用的数据版本
    stage1_mimic_cxr_cache_prefix: str = (
        "mimic_cxr_train_clean_v2_screen160k_seed2048"
    )

    # True：只评测 final_weights
    # False：评测 checkpoint-*，并在最后评测 final_weights
    eval_final_weights_only: bool = True #False

    # Stage 2 是否为 DLR 实验。
    # 评测阶段不会创建优化器，但该字段决定 Stage 2 目录名称。
    #A0为False
    use_discriminative_lr: bool = False #True
    lr_recipe_name: str = "DLR"

    # 所加载的 Stage 1 是否为 DLR 实验。
    # 必须与真实 Stage 1 checkpoint 目录一致。
    #A0为False
    stage1_use_discriminative_lr: bool = False #True
    stage1_lr_recipe_name: str = "DLR"

    # =========================================================
    # 2. 评测结果根目录
    # =========================================================
    eval_output_root: str = (
        "/home/yuqing/Models/MoRA_Med/Eval_VQA_RAD"
    )

    # =========================================================
    # 3. 文本生成参数
    # =========================================================
    max_new_tokens: int = 64

    # VQA-RAD 正式评测使用确定性贪心解码
    do_sample: bool = False
    temperature: float = 0.0

    # =========================================================
    # 4. Eval DataLoader
    # =========================================================
    per_device_eval_batch_size: int = 4
    dataloader_num_workers: int = 4

    # =========================================================
    # 5. __post_init__ 生成的评测路径
    # 不允许在初始化参数中手动传入
    # =========================================================
    stage2_run_dir: str = field(
        init=False,
        default="",
    )

    stage2_weights_dir: str = field(
        init=False,
        default="",
    )

    def __post_init__(self):
        # -----------------------------------------------------
        # 第一步：调用新版 VQA-RAD TrainConfig
        #
        # 这一步会统一完成：
        # - A0～A6 配置绑定
        # - enable_visual_adapter 设置
        # - Conv2D F3/F5/F7 配置
        # - Router/Gate/lambda/RMS 配置
        # - Stage 1 权重目录生成
        # - Stage 2 训练输出目录生成
        # -----------------------------------------------------
        super().__post_init__()

        # 此时 self.output_dir 是 TrainConfig 生成的
        # Stage 2 训练实验目录，必须先保存下来。
        self.stage2_run_dir = self.output_dir

        # 正式评测默认读取最终权重。
        self.stage2_weights_dir = os.path.join(
            self.stage2_run_dir,
            "final_weights",
        )

        # 使用完整 Stage 2 实验目录名作为评测子目录名，
        # 避免 A0、A6、不同 seed 或不同 DLR 实验互相覆盖。
        run_name = os.path.basename(
            os.path.normpath(self.stage2_run_dir)
        )

        # 从这里开始，self.output_dir 表示评测结果目录，
        # 不再表示训练目录。
        self.output_dir = os.path.join(
            self.eval_output_root,
            run_name,
        )

        print("\n" + "=" * 70)
        print(
            "VQA-RAD Stage 2 评测配置初始化完成"
        )
        print(
            f"消融实验：{self.ablation_id}"
        )
        print(
            f"启用视觉 Adapter："
            f"{self.enable_visual_adapter}"
        )
        print(
            f"Stage 2 DLR："
            f"{self.use_discriminative_lr}"
        )
        print(
            f"Stage 1 DLR："
            f"{self.stage1_use_discriminative_lr}"
        )
        print(
            f"Stage 2 训练目录："
            f"{self.stage2_run_dir}"
        )
        print(
            f"Stage 2 最终权重："
            f"{self.stage2_weights_dir}"
        )
        print(
            f"评测结果目录："
            f"{self.output_dir}"
        )
        print("=" * 70 + "\n")
