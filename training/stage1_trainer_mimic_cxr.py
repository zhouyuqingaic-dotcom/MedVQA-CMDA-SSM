"""
CM-DA-SSM Stage 1 · MIMIC-CXR / IMPRESSION

对应论文最终消融计划：
    A0        : Qwen3-VL + LoRA，不挂视觉 Adapter
    A1        : 4D-SSM + Uniform Fusion + Fixed λ + Capped RMS
    A2        : 4D-SSM + CM-RADF + Fixed λ + Capped RMS
    A3        : 4D-SSM + CM-RADF + Learnable λ + Capped RMS
    A3-NoRMS  : 4D-SSM + CM-RADF + Learnable λ + ρ≡1

执行原则：
    1. 五组从相同 Qwen3-VL 基座开始各自独立的 Stage 1；
    2. A0/A1 不加载 BioMedCLIP；A2/A3/A3-NoRMS 才加载；
    3. 对所有样本监督报告 IMPRESSION，不向 BioMedCLIP 泄露目标；
    4. 全部组只增强 pooled_tokens，DeepStack 由 Qwen 原生输出；
    5. 用新版优化器严格验证每组的可训练参数；
    6. 每个 Trainer checkpoint 保存 LoRA + Visual Adapter + manifest；
    7. 可精确断点续训，禁止跨组/跨 seed/跨 run_tag 恢复；
    8. 最终权重保存到 cfg.final_weights_dir，供四套 Stage 2 继承。

运行示例：
    torchrun --nproc_per_node=4 training/stage1_trainer_mimic_cxr.py \\
        --ablation_id A3 --seed 2048 --run_tag main

    torchrun --nproc_per_node=4 training/stage1_trainer_mimic_cxr.py \\
        --ablation_id A3 --seed 2048 --run_tag main \\
        --resume_from_checkpoint auto

注意：
    config.stage1_train_config_mimic_cxr 会解析消融组/seed/run_tag；
    本文件只额外解析 --resume_from_checkpoint，未知 CLI 字段
    交给其它配置解析函数，避免 torchrun LOCAL_RANK 冲突。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# 从项目根目录执行 `python training/...py` 或 `torchrun ... training/...py`
# 时，Python 默认可能只把 training/ 加入 sys.path。
# 在这里显式增加项目根目录，避免导入 config/、datas/、utils/ 失败。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist

from config.stage1_train_config_mimic_cxr import TrainConfig
from utils.ddp.ddp_utils import ddp_print
from utils.training.stage1_checkpoint_utils import (
    create_stage1_checkpoint_callback,
    fingerprint_train_dataset,
    is_global_main_process,
    resolve_resume_checkpoint,
    restore_visual_adapter_strict,
    save_stage1_artifacts,
    validate_stage1_artifacts,
)
from utils.training.discriminative_optimizer import (
    build_discriminative_adamw,
    format_optimizer_groups,
)


def parse_stage1_trainer_options():
    """
    只负责 Trainer 专用命令行。

    用户的 --ablation_id / --seed / --run_tag 已由 TrainConfig 解析，
    此处不能再声明不同默认值或单独覆盖它们。
    """
    parser = argparse.ArgumentParser(
        description="CM-DA-SSM Stage 1 MIMIC-CXR Trainer",
        add_help=True,
    )
    parser.add_argument(
        "--check_config_only",
        action="store_true",
        help=(
            "只打印五组中当前消融的架构/输出路径/参数组学习率，"
            "不加载 8B Qwen、不加载图像、不启动训练。"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "不传表示全新训练；auto 自动选择本实验最后一个完整 checkpoint；"
            "也可以传当前 output_dir/checkpoint-N 的明确路径。"
        ),
    )

    options, _ = parser.parse_known_args()
    return options


def set_seed(seed: int):
    """
    初始化 Python / NumPy / PyTorch 随机种子。

    在模型和 Adapter 实例化之前设置，保证 A3 与 A3-NoRMS
    从相同的随机初值开始（两组权重不能互相继承）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_distributed() -> tuple[int, int, int]:
    """
    兼容 python 单进程和 torchrun 的进程启动方式。

    在构建 MIMIC-CXR indices cache 之前显式初始化 process group，
    使 rank 0 能先生成数据缓存，然后其他 rank 再读取。
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if local_rank >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "检测到 torchrun，但当前无 CUDA 设备。"
                "正式 Qwen3-VL-8B 训练应使用 CUDA 环境。"
            )
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return local_rank, rank, world_size


def rank_barrier():
    """仅在 DDP 已初始化时同步，单 GPU 无需操作。"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def check_same_dataset_across_ranks(digest: str) -> None:
    """
    各 DDP 进程必须加载完全相同、顺序一致的 MIMIC-CXR 样本子集。

    只比较固定长度的 SHA256，不同步医学图像或报告内容。
    """
    if dist.is_available() and dist.is_initialized():
        digests = [None] * dist.get_world_size()
        dist.all_gather_object(digests, digest)

        if len(set(digests)) != 1:
            raise RuntimeError(
                "不同 DDP rank 的 MIMIC-CXR 样本指纹不一致，"
                "请检查索引缓存或数据集路径。"
            )


def build_train_dataset(
    cfg: TrainConfig,
    *,
    allow_cache_rebuild: bool = True,
):
    """
    用原工程 MIMICCXRDataset 构造数据集，不更改过滤/缓存逻辑。

    不允许按 ablation_id 改变数据内容、缓存策略、样本数量：
    五组应共享同一份 160k MIMIC-CXR 子集以保证消融公平。
    """
    from datas.mimic_cxr_datasets import MIMICCXRDataset

    dataset = MIMICCXRDataset(
        csv_path=cfg.mimic_cxr_metadata_csv,
        image_root=cfg.mimic_cxr_image_root,
        report_root=cfg.mimic_cxr_report_root,
        target_section=cfg.mimic_cxr_target_section,
        load_report=cfg.mimic_cxr_load_report,
        allowed_view_positions=cfg.mimic_cxr_view_positions,
        drop_empty_target=cfg.mimic_cxr_drop_empty_target,
        cache_dir=cfg.mimic_cxr_cache_dir,
        use_indices_cache=cfg.mimic_cxr_use_indices_cache,
        # 即使用户显式开启 rebuild，非 rank0 也必须跳过重建；
        # 避免 rank0 写完后其他进程重复扫描并并发覆盖缓存文件。
        rebuild_indices_cache=(
            cfg.mimic_cxr_rebuild_indices_cache and allow_cache_rebuild
        ),
        cache_prefix=cfg.mimic_cxr_cache_prefix,
    )

    if cfg.max_mimic_cxr_train_samples is not None:
        dataset.samples = dataset.samples[
            :cfg.max_mimic_cxr_train_samples
        ]

    if len(dataset) <= 0:
        raise RuntimeError("当前 MIMIC-CXR 训练集为空。")

    return dataset


def load_dataset_rank0_first(cfg: TrainConfig, rank: int):
    """
    避免多个 torchrun 进程同时写 MIMIC-CXR 索引缓存。

    MIMICCXRDataset 的旧缓存保存使用 json.dump，没有文件锁；
    所以先在 global rank 0 生成缓存，其他 rank 随后只读。
    """
    dataset = None

    if rank == 0:
        dataset = build_train_dataset(cfg)

    rank_barrier()

    if rank != 0:
        dataset = build_train_dataset(
            cfg,
            allow_cache_rebuild=False,
        )

    rank_barrier()
    return dataset


def load_base_and_biomed(cfg: TrainConfig, local_rank: int):
    """
    只从 cfg.model_name_or_path 加载同一个 Qwen3-VL 预训练基座。

    BioMedCLIP 严格根据 cfg.requires_biomedclip 决定：
        A0/A1 -> None
        A2/A3/A3-NoRMS -> 冻结的图像/文本编码器与预处理工具

    A1 的 Uniform Fusion 不依赖 Query，因此无须加载 BioMedCLIP。
    """
    # 延迟导入：便于在只装了 PyTorch 的环境测试配置和 checkpoint。
    from utils.qwen3vl.qwen3_vl_8B_quant_loader import (
        Qwen3VLQuantizedLoader,
    )

    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map=(
            {"": local_rank}
            if local_rank != -1
            else "auto"
        ),
    )

    base_model, processor = loader.load()
    processor.tokenizer.padding_side = "right"

    biomed_extractor = None
    biomed_transform = None
    biomed_tokenizer = None

    if cfg.requires_biomedclip:
        from utils.biomedclip.biomed_clip_loader import load_biomedclip

        (
            biomed_extractor,
            biomed_transform,
            biomed_tokenizer,
        ) = load_biomedclip(
            biomedclip_path=cfg.biomedclip_path,
            print_rank=cfg.print_rank,
        )

    return (
        base_model,
        processor,
        biomed_extractor,
        biomed_transform,
        biomed_tokenizer,
    )


def wrap_stage1_model(cfg: TrainConfig, base_model, biomed_extractor):
    """
    直接使用新配置生成 Wrapper 参数，避免复制旧 Gate/Router 字段。

    关键：
        cfg.wrapper_kwargs(...) 根据 ablation_id 得到实际开关，
        cfg.ablation_rule 可用于记录日志和验证。
    """
    from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_lora_wrapper import (
        Qwen3VLLoraAndCMDASSMWrapper,
    )

    wrapper = Qwen3VLLoraAndCMDASSMWrapper(
        **cfg.wrapper_kwargs(
            biomed_extractor=biomed_extractor,
        )
    )

    return wrapper.wrap(base_model)


def build_stage1_collator(
    cfg: TrainConfig,
    processor,
    biomed_transform=None,
    biomed_tokenizer=None,
):
    """
    A0/A1 不需要 BiomedCLIP，Collator 不应产出额外的图文张量。
    A2/A3/A3-NoRMS 采用固定报告指令的 BiomedCLIP Text Tokens。
    """
    from utils.data_tools.collator.mimic_cxr.mimic_cxr_datasets_train_collator import (
        MIMICCXRTrainCollator,
    )

    return MIMICCXRTrainCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_transform,
        biomed_tokenizer=biomed_tokenizer,
    )


def build_training_arguments(cfg: TrainConfig):
    """
    保留原 Stage 1 的 HF Trainer 主流程与训练参数。

    只有分组优化器由新版 build_discriminative_adamw 创建；
    Trainer 继续自动创建 Scheduler 并管理混合精度/DDP/梯度累积。

    save_only_model=False 是精确断点恢复的必要条件：
    checkpoint 必须保留 optimizer/scheduler/RNG。
    """
    from transformers import TrainingArguments

    is_bf16 = cfg.torch_dtype in ("bfloat16", torch.bfloat16)
    is_fp16 = cfg.torch_dtype in ("float16", torch.float16)

    return TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,

        learning_rate=cfg.lora_learning_rate,
        weight_decay=cfg.weight_decay,
        optim="adamw_torch",

        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,

        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_only_model=False,

        bf16=is_bf16,
        fp16=is_fp16,
        dataloader_num_workers=cfg.dataloader_num_workers,

        # LoRA Wrapper 已调用 prepare_model_for_kbit_training，
        # 这里不重复启用 HF 的梯度检查点包装。
        gradient_checkpointing=False,

        # A2/A3/A3-NoRMS 的 biomed_image_tensors/tokens 是 Qwen
        # 原始 forward 未声明的额外字段，不能被 Trainer 自动删除。
        remove_unused_columns=False,

        # 本项目各分支的全部 trainable 参数都在计算图中；
        # A1 并未创建完全无用的 CM-RADF Query/Key。
        ddp_find_unused_parameters=False,

        # 避免引入无关实验追踪平台影响比较。
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
    )


def create_stage1_trainer(
    *,
    cfg: TrainConfig,
    model,
    train_dataset,
    collator,
    training_args,
    train_sample_digest: str | None = None,
):
    """
    配置新的分组优化器与 checkpoint callback。

    使用 cfg.optimizer_kwargs(training_args) 避免人工重复配置
    旧 MoRA-Med 的 visual_expert/router_gate 参数。
    """
    from transformers import Trainer

    optimizer = build_discriminative_adamw(
        model,
        **cfg.optimizer_kwargs(training_args),
    )

    ddp_print(
        format_optimizer_groups(optimizer),
        print_rank=cfg.print_rank,
    )

    callback = create_stage1_checkpoint_callback(
        cfg=cfg,
        train_sample_count=len(train_dataset),
        train_sample_digest=train_sample_digest,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        optimizers=(optimizer, None),
        callbacks=[callback],
    )


def save_final_stage1(
    *,
    trainer,
    model,
    processor,
    cfg: TrainConfig,
    train_sample_count: int,
    train_sample_digest: str | None = None,
) -> None:
    """
    Stage 1 训练结束后完整保存：
        final_weights/adapter_model.safetensors
        final_weights/adapter_config.json
        final_weights/visual_adapter.pt  (仅 A1/A2/A3/A3-NoRMS)
        final_weights/architecture.json
        final_weights/processor...       (Qwen Processor 文件)

    为兼容 DDP/Accelerate，trainer.save_model 必须所有 rank 都调用；
    它自己决定哪个进程实际写 PEFT LoRA。

    只有 global rank 0 写 Processor、视觉 Adapter 和架构清单。
    """
    final_dir = Path(cfg.final_weights_dir)

    # 所有分布式进程都经过 HF Trainer.save_model，
    # 避免某些并行策略只在 rank0 调用时出现同步死锁。
    trainer.save_model(str(final_dir))

    if is_global_main_process(trainer):
        final_dir.mkdir(parents=True, exist_ok=True)

        # Stage 2 也需要 Qwen Processor，保持与原工程的加载方式一致。
        processor.save_pretrained(str(final_dir))

        manifest = save_stage1_artifacts(
            model=model,
            cfg=cfg,
            directory=final_dir,
            global_step=int(trainer.state.global_step),
            train_sample_count=train_sample_count,
            train_sample_digest=train_sample_digest,
            is_final=True,
        )

        # 保存完立即重新检查 SHA256 和配置，防止产出不可用权重。
        validate_stage1_artifacts(
            cfg=cfg,
            directory=final_dir,
            expected_train_sample_count=train_sample_count,
            expected_train_sample_digest=train_sample_digest,
            require_optimizer_state=False,
        )

        ddp_print(
            "\n🎉 Stage 1 完成，已保存完整的 LoRA + 视觉 Adapter "
            "+ architecture.json。",
            print_rank=cfg.print_rank,
        )
        ddp_print(
            f"    Ablation : {manifest['ablation_id']}",
            f"\n    Step     : {manifest['global_step']}",
            f"\n    Location : {final_dir}",
            print_rank=cfg.print_rank,
        )

    rank_barrier()


def main():
    options = parse_stage1_trainer_options()
    cfg = TrainConfig()

    if options.check_config_only:
        # 配置预检：无需 CUDA、PEFT、Mamba 或本地医学数据。
        # 可在提交正式 4GPU 任务之前核对实验组及输出目录。
        ddp_print(
            "CM-DA-SSM Stage 1 configuration preview",
            print_rank=cfg.print_rank,
        )
        ddp_print(
            f"  Ablation: {cfg.ablation_id} | "
            f"Fusion: {cfg.fusion_mode} | "
            f"Lambda: {cfg.lambda_mode} | "
            f"RMS: {cfg.use_rms_norm}",
            print_rank=cfg.print_rank,
        )
        ddp_print(
            f"  BiomedCLIP required: {cfg.requires_biomedclip}",
            print_rank=cfg.print_rank,
        )
        ddp_print(
            f"  Architecture: {cfg.architecture_tag}",
            print_rank=cfg.print_rank,
        )
        ddp_print(
            f"  Output: {cfg.final_weights_dir}",
            print_rank=cfg.print_rank,
        )
        ddp_print(
            f"  Optimizer kwargs: {cfg.optimizer_kwargs()}",
            print_rank=cfg.print_rank,
        )
        return

    local_rank, rank, world_size = initialize_distributed()

    try:
        set_seed(cfg.seed)

        ddp_print(
            "\n" + "=" * 72,
            print_rank=cfg.print_rank,
        )
        ddp_print(
            "🚀 CM-DA-SSM Stage 1 · MIMIC-CXR / IMPRESSION",
            print_rank=cfg.print_rank,
        )
        ddp_print(
            f"    Ablation={cfg.ablation_id}, "
            f"Seed={cfg.seed}, WorldSize={world_size}, "
            f"RunTag={cfg.run_tag}",
            print_rank=cfg.print_rank,
        )
        ddp_print(
            f"    Fusion={cfg.fusion_mode}, "
            f"Lambda={cfg.lambda_mode}, "
            f"RMS={cfg.use_rms_norm}, "
            f"BiomedCLIP={cfg.requires_biomedclip}, "
            f"DeepStack=False",
            print_rank=cfg.print_rank,
        )
        ddp_print("=" * 72, print_rank=cfg.print_rank)

        # =============================================================
        # Step 0. 训练前的路径防覆盖检查
        # =============================================================
        # 在加载 8B 模型之前阻止误读 A3/A3-NoRMS 等其他组的权重。
        resume_dir = resolve_resume_checkpoint(
            cfg=cfg,
            output_dir=cfg.output_dir,
            requested=options.resume_from_checkpoint,
        )

        if rank == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)

        rank_barrier()

        # =============================================================
        # Step 1. MIMIC-CXR Dataset
        # =============================================================
        train_dataset = load_dataset_rank0_first(
            cfg,
            rank=rank,
        )

        ddp_print(
            f"✅ 已加载 MIMIC-CXR 训练样本：{len(train_dataset):,}",
            print_rank=cfg.print_rank,
        )

        # 不只比较样本数，还对样本内容和顺序生成 SHA256。
        # 所有 DDP rank 检查同一份子集；不会把医学报告写进日志。
        train_sample_digest = fingerprint_train_dataset(
            train_dataset
        )
        check_same_dataset_across_ranks(train_sample_digest)

        ddp_print(
            f"    Dataset fingerprint: {train_sample_digest[:16]}",
            print_rank=cfg.print_rank,
        )

        # 如果恢复训练，还要确认样本数及顺序完全匹配。
        if resume_dir is not None:
            validate_stage1_artifacts(
                cfg=cfg,
                directory=resume_dir,
                expected_train_sample_count=len(train_dataset),
                expected_train_sample_digest=train_sample_digest,
                require_optimizer_state=True,
            )

        # =============================================================
        # Step 2. 载入同一个 Qwen3-VL 基座与所需 BioMedCLIP
        # =============================================================
        (
            base_model,
            processor,
            biomed_extractor,
            biomed_transform,
            biomed_tokenizer,
        ) = load_base_and_biomed(
            cfg=cfg,
            local_rank=local_rank,
        )

        # =============================================================
        # Step 3. 注入当前消融组的 LoRA + CM-DA-SSM
        # =============================================================
        model = wrap_stage1_model(
            cfg=cfg,
            base_model=base_model,
            biomed_extractor=biomed_extractor,
        )

        # =============================================================
        # Step 4. 只恢复视觉适配器的完整权重
        # =============================================================
        # 若没有 resume，保留 Shared Up=0 的初始恒等状态；
        # 若有 resume，必须在 Trainer 开始恢复 optimizer 前执行。
        #
        # LoRA/optimizer/scheduler/RNG 的恢复由 HF Trainer 完成。
        if resume_dir is not None:
            restore_visual_adapter_strict(
                model,
                cfg=cfg,
                directory=resume_dir,
            )

            ddp_print(
                f"♻️ 正在恢复 Stage 1：{resume_dir}",
                print_rank=cfg.print_rank,
            )

        # =============================================================
        # Step 5. Collator（A0/A1 不生成 BiomedCLIP 张量）
        # =============================================================
        collator = build_stage1_collator(
            cfg=cfg,
            processor=processor,
            biomed_transform=biomed_transform,
            biomed_tokenizer=biomed_tokenizer,
        )

        # =============================================================
        # Step 6. TrainingArguments + 分组 AdamW + Checkpoint Callback
        # =============================================================
        args = build_training_arguments(cfg)

        trainer = create_stage1_trainer(
            cfg=cfg,
            model=model,
            train_dataset=train_dataset,
            collator=collator,
            training_args=args,
            train_sample_digest=train_sample_digest,
        )

        # =============================================================
        # Step 7. 正式训练（含精确断点续训）
        # =============================================================
        # 不设置 ignore_data_skip=True，确保数据采样位置也继续恢复。
        trainer.train(
            resume_from_checkpoint=resume_dir,
        )

        # =============================================================
        # Step 8. 保存最终 Stage 1 结果供 Stage 2 继承
        # =============================================================
        save_final_stage1(
            trainer=trainer,
            model=model,
            processor=processor,
            cfg=cfg,
            train_sample_count=len(train_dataset),
            train_sample_digest=train_sample_digest,
        )

    finally:
        # 如 torchrun 已初始化 DDP，则训练完成/异常退出时释放资源。
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
