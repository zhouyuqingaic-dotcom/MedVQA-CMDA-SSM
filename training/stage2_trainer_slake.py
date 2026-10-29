import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset,Subset  # 🚀 降维打击核心工具：缝合数据集
from transformers import Trainer, TrainingArguments, TrainerCallback

# 1. 导入 SLAKE 专属配置类
from config.slake.stage2_train_config_slake import Stage2TrainConfig
# 2. 导入 SLAKE 数据集
from datas.slake_datasets import SLAKEDataset
# 3. 导入 SLAKE 专属 Collator
from utils.data_tools.collator.slake.slake_datasets_train_collator import SLAKETrainCollator

from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader

# 🚀 终极完全体 Wrapper 与 BioMedCLIP 加载器
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import Qwen3VLLoraAndVisualAdapterWrapper
from utils.biomedclip.biomed_clip_loader import load_biomedclip
from utils.ddp.ddp_utils import ddp_print
from utils.training.discriminative_optimizer import (
    build_discriminative_adamw,
    format_optimizer_groups,
)

# 引入 PEFT 的状态字典注入工具
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from utils.data_tools.prompt_cleaning.slake_answer_cleaning import slake_answer_train_cleaning

def filter_invalid_slake_samples(
    dataset,
    split_name: str,
    print_rank: int,
):
    """
    通过 Dataset.__getitem__ 检查训练时实际返回的样本，
    并使用 Subset 保留具有有效监督信号的样本。

    这样过滤的对象与 DataLoader、Collator 实际读取的对象完全一致。
    """
    original_count = len(dataset)

    valid_indices = []
    invalid_samples = []

    for index in range(original_count):
        # 通过 __getitem__ 获取与 Collator 完全相同的数据结构。
        sample = dataset[index]

        question_text = str(
            sample.get("question", "")
        ).strip()

        answer_source = str(
            sample.get("answer", "")
        ).strip()

        answer_text = slake_answer_train_cleaning(
            answer_source
        )

        if question_text and answer_text:
            valid_indices.append(index)
            continue

        invalid_samples.append(
            {
                "index": index,
                "image_path": sample.get("image_path", ""),
                "question": question_text,
                "raw_answer": repr(answer_source),
                "cleaned_answer": repr(answer_text),
            }
        )

    filtered_dataset = Subset(
        dataset,
        valid_indices,
    )

    ddp_print(
        f"[SLAKE 数据清理] {split_name}: "
        f"原始={original_count}, "
        f"保留={len(filtered_dataset)}, "
        f"删除={len(invalid_samples)}",
        print_rank=print_rank,
    )

    if invalid_samples:
        preview = invalid_samples[:10]

        ddp_print(
            f"[SLAKE 数据清理] {split_name} "
            "无效样本示例（最多 10 条）：\n"
            + "\n".join(
                (
                    f"  index={item['index']}, "
                    f"image_path={item['image_path']}, "
                    f"question={item['question']!r}, "
                    f"raw_answer={item['raw_answer']}, "
                    f"cleaned_answer={item['cleaned_answer']}"
                )
                for item in preview
            ),
            print_rank=print_rank,
        )

    return filtered_dataset



def get_visual_adapter(model):
    """
    获取 RoMA-Net V2-lite visual adapter。

    兼容普通 PEFT 模型和 DDP 包装后的模型。
    """
    if hasattr(model, "module"):
        model = model.module

    try:
        return model.base_model.model.model.visual.res_adapter
    except AttributeError:
        return None


class VisualAdapterSaveCallback(TrainerCallback):
    """
    在 Hugging Face Trainer 保存 checkpoint 时，
    同步保存 RoMA-Net visual adapter。

    A0 不包含 visual adapter，因此自动跳过。
    A1～A6 会在 checkpoint-* 中保存 visual_adapter.pt。
    """

    def __init__(self, enable_visual_adapter: bool):
        self.enable_visual_adapter = bool(enable_visual_adapter)

    def on_save(self, args, state, control, **kwargs):
        # A0 不包含 visual adapter。
        if not self.enable_visual_adapter:
            return control

        # 只允许主进程写文件，避免多卡同时覆盖。
        if not state.is_world_process_zero:
            return control

        model = kwargs["model"]
        adapter_module = get_visual_adapter(model)

        if adapter_module is None:
            raise RuntimeError(
                "enable_visual_adapter=True，"
                "但模型中没有找到 visual.res_adapter。"
            )

        checkpoint_dir = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        adapter_save_path = os.path.join(
            checkpoint_dir,
            "visual_adapter.pt",
        )

        # 转到 CPU 后保存，避免保存 GPU Tensor。
        adapter_state_dict = {
            name: tensor.detach().cpu()
            for name, tensor in adapter_module.state_dict().items()
        }

        torch.save(
            adapter_state_dict,
            adapter_save_path,
        )

        print(
            "\n[Callback] Visual adapter 已保存至："
            f"{adapter_save_path}"
        )

        return control




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # ==========================================
    # 0. DDP 环境感知与初始化
    # ==========================================
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)

    cfg = Stage2TrainConfig()
    output_dir = cfg.output_dir
    stage1_weights_dir = cfg.stage1_weights_dir

    set_seed(cfg.seed)

    ddp_print("\n" + "=" * 60, print_rank=cfg.print_rank)
    ddp_print(
        f"🚀 [1/6] 启动 RoMA-Net V2-lite Stage 2 SLAKE 训练！"
        f"当前消融: {cfg.ablation_id}",
        print_rank=cfg.print_rank,
    )
    ddp_print(
        f"    enable_visual_adapter={cfg.enable_visual_adapter}, "
        f"scale_mode={cfg.scale_mode}, "
        f"gate_mode={cfg.gate_mode}, "
        f"lambda_mode={cfg.lambda_mode}, "
        f"use_rms_norm={cfg.use_rms_norm}",
        print_rank=cfg.print_rank,
    )
    ddp_print(
        f"🔗 Stage 1 权重目录: {stage1_weights_dir}",
        print_rank=cfg.print_rank,
    )
    ddp_print(
        f"📁 Stage 2 输出目录: {output_dir}",
        print_rank=cfg.print_rank,
    )
    ddp_print("=" * 60, print_rank=cfg.print_rank)


    if local_rank in [-1, 0]:
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(stage1_weights_dir):
            raise FileNotFoundError(f"❌ 找不到 Stage 1 权重，请核实路径: {stage1_weights_dir}")

    # ==========================================
    # 1. 🚀 核心修改：加载并无缝缝合 SLAKE Train + Val 数据集
    # ==========================================
    ddp_print("⏳ [2/6] 正在加载并合并 SLAKE Train 和 Val 数据集...", print_rank=cfg.print_rank)

    train_subset = SLAKEDataset(
        json_path=cfg.slake_train_json_path,
        image_root=cfg.slake_image_root,
    )

    val_subset = SLAKEDataset(
        json_path=cfg.slake_val_json_path,
        image_root=cfg.slake_image_root,
    )

    # 施加黑魔法：拼装成将近 1.2 万条数据的训练集
    # 在构造 ConcatDataset 和 DDP DataLoader 前过滤无效监督样本。
    train_subset = filter_invalid_slake_samples(
        dataset=train_subset,
        split_name="train",
        print_rank=cfg.print_rank,
    )

    val_subset = filter_invalid_slake_samples(
        dataset=val_subset,
        split_name="validate",
        print_rank=cfg.print_rank,
    )

    train_dataset = ConcatDataset(
        # [train_subset, val_subset]
        [train_subset]
    )

    ddp_print(
        f"✅ 数据集清理与合并完成，共有 "
        f"{len(train_dataset)} 条有效医学样本。",
        print_rank=cfg.print_rank,
    )

    # ==========================================
    # 2. 加载 Qwen3-VL 4-bit 底座模型
    # ==========================================
    ddp_print("\n⏳ [3/6] 正在加载 Qwen3-VL 4-bit 底座模型...", print_rank=cfg.print_rank)
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map={"": local_rank} if local_rank != -1 else "auto",
    )
    base_model, processor = loader.load()
    processor.tokenizer.padding_side = "right"
    ddp_print("✅ 底座加载完毕！", print_rank=cfg.print_rank)

    # ==========================================
    # 3. 根据 enable_visual_adapter 初始化 BioMedCLIP
    # ==========================================
    biomed_extractor = None
    biomed_transform = None
    biomed_tokenizer = None

    if cfg.enable_visual_adapter:
        ddp_print(
            "\n⏳ [4/6] 正在加载 BioMedCLIP...",
            print_rank=cfg.print_rank,
        )

        biomed_extractor, biomed_transform, biomed_tokenizer = load_biomedclip(
            biomedclip_path=cfg.biomedclip_path,
            print_rank=cfg.print_rank,
        )

        ddp_print(
            "✅ BioMedCLIP 加载完成。",
            print_rank=cfg.print_rank,
        )
    else:
        ddp_print(
            "\nℹ️ 当前为 A0 / LoRA-only，不加载 BioMedCLIP。",
            print_rank=cfg.print_rank,
        )

    # ==========================================
    # 4. 🎯 模型接驳与 Stage 1 完美夺舍
    # ==========================================
    ddp_print("\n⏳ [4/6] 正在执行模型接驳与 Stage 1 权重继承...", print_rank=cfg.print_rank)

    # 4.1 使用 Wrapper 组装架构
    wrapper = Qwen3VLLoraAndVisualAdapterWrapper(
        # LoRA
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        gradient_checkpointing=cfg.gradient_checkpointing,

        # Visual adapter
        visual_adapter_hidden_dim=cfg.visual_adapter_hidden_dim,
        visual_adapter_r=cfg.visual_adapter_r,
        enable_visual_adapter=cfg.enable_visual_adapter,

        # BioMedCLIP route feature
        biomed_extractor=biomed_extractor,
        use_cross_modal_prior=cfg.use_cross_modal_prior,

        # Router backbone
        router_hidden_dim=cfg.router_hidden_dim,

        # 三尺度 Conv2D 路由：
        # index 0 -> F3 / 3x3
        # index 1 -> F5 / 5x5
        # index 2 -> F7 / 7x7
        scale_mode=cfg.scale_mode,
        fixed_scale_weights=cfg.fixed_scale_weights,

        # Soft gate
        gate_mode=cfg.gate_mode,
        fixed_gate=cfg.fixed_gate,
        gate_init=cfg.gate_init,

        # 有界 lambda
        lambda_mode=cfg.lambda_mode,
        fixed_lambda=cfg.fixed_lambda,
        lambda_max=cfg.lambda_max,
        lambda_init=cfg.lambda_init,

        # RMS residual normalization
        use_rms_norm=cfg.use_rms_norm,
        residual_norm_eps=cfg.residual_norm_eps,
        residual_norm_ratio_clip=cfg.residual_norm_ratio_clip,
    )

    ddp_print(
        "    visual_experts=DWConv2D F3/F5/F7 "
        "(3x3 / 5x5 / 7x7)",
        print_rank=cfg.print_rank,
    )

    peft_model = wrapper.wrap(base_model)

    # 4.2 🚀 注入 Stage 1 的 LoRA 权重
    lora_safe_path = os.path.join(
        stage1_weights_dir,
        "adapter_model.safetensors",
    )

    lora_bin_path = os.path.join(
        stage1_weights_dir,
        "adapter_model.bin",
    )

    if os.path.isfile(lora_safe_path):
        lora_state_dict = load_file(lora_safe_path)

        set_peft_model_state_dict(
            peft_model,
            lora_state_dict,
        )

        ddp_print(
            f"✅ 已加载 Stage 1 LoRA：{lora_safe_path}",
            print_rank=cfg.print_rank,
        )

    elif os.path.isfile(lora_bin_path):
        lora_state_dict = torch.load(
            lora_bin_path,
            map_location="cpu",
        )

        set_peft_model_state_dict(
            peft_model,
            lora_state_dict,
        )

        ddp_print(
            f"✅ 已加载 Stage 1 LoRA：{lora_bin_path}",
            print_rank=cfg.print_rank,
        )

    else:
        raise FileNotFoundError(
            "找不到 Stage 1 LoRA 权重：\n"
            f"  {lora_safe_path}\n"
            f"  {lora_bin_path}"
        )

    # 4.3 🚀 注入 Stage 1 的 MoE 视觉适配器权重
    if cfg.enable_visual_adapter:
        adapter_pt_path = os.path.join(
            stage1_weights_dir,
            "visual_adapter.pt",
        )

        if not os.path.isfile(adapter_pt_path):
            raise FileNotFoundError(
                "当前消融启用了 visual adapter，"
                "但找不到 Stage 1 权重："
                f"{adapter_pt_path}"
            )

        adapter_module = get_visual_adapter(peft_model)

        if adapter_module is None:
            raise RuntimeError(
                "当前配置 enable_visual_adapter=True，"
                "但模型中没有成功挂载 visual.res_adapter。"
            )

        adapter_state_dict = torch.load(
            adapter_pt_path,
            map_location="cpu",
        )

        # Stage 2 必须与 Stage 1 采用完全一致的架构。
        # strict=True 可在配置不一致时立即报错。
        adapter_module.load_state_dict(
            adapter_state_dict,
            strict=True,
        )


        ddp_print(
            f"✅ 已加载 Stage 1 visual adapter：{adapter_pt_path}",
            print_rank=cfg.print_rank,
        )

    else:
        ddp_print(
            "ℹ️ 当前为 A0 / LoRA-only，"
            "只加载 Stage 1 LoRA，不加载 visual_adapter.pt。",
            print_rank=cfg.print_rank,
        )

    peft_model.print_trainable_parameters()

    # ==========================================
    # 5. 🚀 挂载 SLAKE 专属 Collator
    # ==========================================
    ddp_print("\n⏳ [5/6] 挂载 SLAKE 专属 Collator...", print_rank=cfg.print_rank)
    collator = SLAKETrainCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_transform,
        biomed_tokenizer=biomed_tokenizer
    )

    # ==========================================
    # 6. 配置 HF TrainingArguments & 启动
    # ==========================================
    ddp_print(f"\n🔥 [6/6] 启动 Hugging Face Trainer... 设定轮数: {cfg.num_train_epochs}", print_rank=cfg.print_rank)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,

        learning_rate=(
            cfg.lora_learning_rate
            if cfg.use_discriminative_lr
            else cfg.learning_rate
        ),
        optim="adamw_torch",

        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=(cfg.torch_dtype == "bfloat16" or cfg.torch_dtype == torch.bfloat16),
        fp16=(cfg.torch_dtype == "float16" or cfg.torch_dtype == torch.float16),
        dataloader_num_workers=cfg.dataloader_num_workers,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",

        # 控制 Trainer 内部训练随机状态
        seed=cfg.seed,

        # 显式控制数据 sampler；和 seed 相同时属于推荐但非必需
        data_seed=cfg.seed,
    )

    trainer_kwargs = {}

    if cfg.use_discriminative_lr:
        optimizer = build_discriminative_adamw(
            peft_model,
            lora_learning_rate=(
                cfg.lora_learning_rate
            ),
            visual_expert_learning_rate=(
                cfg.visual_expert_learning_rate
            ),
            router_gate_learning_rate=(
                cfg.router_gate_learning_rate
            ),
            visual_norm_learning_rate=(
                cfg.visual_norm_learning_rate
            ),
            weight_decay=cfg.weight_decay,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
        )

        trainer_kwargs["optimizers"] = (
            optimizer,
            None,
        )

        ddp_print(
            format_optimizer_groups(optimizer),
            print_rank=cfg.print_rank,
        )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[
            VisualAdapterSaveCallback(
                enable_visual_adapter=(
                    cfg.enable_visual_adapter
                )
            )
        ],
        **trainer_kwargs,
    )

    trainer.train()

    # ==========================================
    # 7. 保存最终权重 (Stage 2 SLAKE 结业)
    # ==========================================
    if local_rank in [-1, 0]:
        final_save_path = os.path.join(output_dir, "final_weights")
        os.makedirs(final_save_path, exist_ok=True)

        trainer.save_model(final_save_path)
        processor.save_pretrained(final_save_path)
        ddp_print(f"\n🎉 Stage 2 (SLAKE) 训练完成！LoRA 已保存至: {final_save_path}", print_rank=cfg.print_rank)

        if cfg.enable_visual_adapter:
            adapter_module = get_visual_adapter(peft_model)

            if adapter_module is None:
                raise RuntimeError(
                    "训练结束时没有找到 visual.res_adapter，"
                    "无法保存 Stage 2 visual adapter。"
                )

            adapter_save_path = os.path.join(
                final_save_path,
                "visual_adapter.pt",
            )

            adapter_state_dict = {
                name: tensor.detach().cpu()
                for name, tensor in adapter_module.state_dict().items()
            }

            torch.save(
                adapter_state_dict,
                adapter_save_path,
            )

            ddp_print(
                f"✨ Stage 2 visual adapter 已保存至: "
                f"{adapter_save_path}",
                print_rank=cfg.print_rank,
            )
        else:
            ddp_print(
                "ℹ️ 当前为 A0 / LoRA-only，"
                "不保存 visual_adapter.pt。",
                print_rank=cfg.print_rank,
            )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()