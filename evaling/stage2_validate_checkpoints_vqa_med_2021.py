import csv
import glob
import json
import math
import os
import types

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.LLM_config import LLMAPIConfig
# VQA-Med 2021 评测配置
from config.vqa_med_2021.stage2_eval_config_vqa_med_2021 import (
    Stage2EvalConfig,
)
# VQA-Med 2021 数据集
from datas.vqa_med_2021_datasets import (
    VQAMED2021Dataset,
)
from LLM_api.deepseek import DeepSeekClient
from utils.biomedclip.biomed_clip_loader import load_biomedclip
# VQA-Med 2021 专属 collator
from utils.data_tools.collator.vqa_med_2021.vqa_med_2021_eval_collator import (
    VQAMED2021EvalCollator,
)
from utils.data_tools.prompt_cleaning.vqa_med_2021_answer_cleaning import (
    vqa_med_2021_answer_eval_cleaning,
    vqa_med_2021_references_eval_cleaning,
)
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import (
    Qwen3VLLoraAndVisualAdapterWrapper,
)
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader


def get_adapter(model):
    """
    获取挂载在 Qwen3-VL 视觉塔中的 Visual Adapter。

    当前模型层级为：
    PEFT model
      -> base_model
      -> model
      -> model
      -> visual
      -> res_adapter
    """
    return model.base_model.model.model.visual.res_adapter


def build_model(weights_path, loader, cfg, biomed_extractor):
    """
    为指定 checkpoint 重建完整模型，并加载：

    1. Qwen3-VL 4-bit 底座；
    2. 当前 Stage 2 checkpoint 的 LoRA；
    3. 当前 Stage 2 checkpoint 的 Visual Adapter。

    每个 checkpoint 都从干净底座重新构建，避免不同 checkpoint
    之间残留参数或状态。
    """
    base_model = loader.load_model()

    # 使用与训练阶段完全一致的 Wrapper 重建模型结构。
    wrapper = Qwen3VLLoraAndVisualAdapterWrapper(
        # LoRA 配置。
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        gradient_checkpointing=False,

        # Visual Adapter 配置。
        visual_adapter_hidden_dim=cfg.visual_adapter_hidden_dim,
        visual_adapter_r=cfg.visual_adapter_r,
        enable_visual_adapter=cfg.enable_visual_adapter,

        # BioMedCLIP 路由条件。
        biomed_extractor=biomed_extractor,
        use_cross_modal_prior=cfg.use_cross_modal_prior,

        # Router 配置。
        router_hidden_dim=cfg.router_hidden_dim,
        scale_mode=cfg.scale_mode,
        fixed_scale_weights=cfg.fixed_scale_weights,

        # 样本级 residual gate。
        gate_mode=cfg.gate_mode,
        fixed_gate=cfg.fixed_gate,
        gate_init=cfg.gate_init,

        # 全局残差系数 lambda。
        lambda_mode=cfg.lambda_mode,
        fixed_lambda=cfg.fixed_lambda,
        lambda_max=cfg.lambda_max,
        lambda_init=cfg.lambda_init,

        # Visual residual RMS 对齐。
        use_rms_norm=cfg.use_rms_norm,
        residual_norm_eps=cfg.residual_norm_eps,
        residual_norm_ratio_clip=cfg.residual_norm_ratio_clip,
    )

    model = wrapper.wrap(base_model)

    # 加载当前 checkpoint 的 LoRA 参数。
    set_peft_model_state_dict(
        model,
        load_file(
            os.path.join(
                weights_path,
                "adapter_model.safetensors",
            )
        ),
    )

    # A0 不启用 Visual Adapter，因此不加载 visual_adapter.pt。
    if cfg.enable_visual_adapter:
        state = torch.load(
            os.path.join(
                weights_path,
                "visual_adapter.pt",
            ),
            map_location="cpu",
        )

        # strict=True：结构或参数名不一致时立即报错。
        get_adapter(model).load_state_dict(
            state,
            strict=True,
        )

    # 评测阶段关闭梯度与 gradient checkpointing，并开启 KV cache。
    model.requires_grad_(False)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    return model, base_model


def patch_generate_forward(model):
    """
    generate() 会连续多次调用 model.forward()。

    BioMedCLIP 特征已经在每个 batch 生成前计算并缓存到 vision_tower，
    因此这里移除原始 BioMedCLIP 输入，避免每生成一个 token
    都重复执行 BioMedCLIP 编码。
    """
    def forward(self, *args, **kwargs):
        kwargs.pop(
            "biomed_image_tensors",
            None,
        )
        kwargs.pop(
            "biomed_text_tokens",
            None,
        )

        # original_forward 是 Wrapper 保存的 PEFT/Qwen 原始 forward。
        return self.original_forward(
            *args,
            **kwargs,
        )

    model.forward = types.MethodType(
        forward,
        model,
    )



def build_llm_judge_user_prompt(
    question,
    references_raw,
    references_norm,
    pred_raw,
    pred_norm,
):
    return (
        f"Question: {str(question).strip()}\n"
        "Raw Acceptable Ground Truth Answers: "
        f"{json.dumps(list(references_raw), ensure_ascii=False)}\n"
        "Normalized Acceptable Ground Truth Answers: "
        f"{json.dumps(list(references_norm), ensure_ascii=False)}\n"
        f"Raw Prediction: {str(pred_raw).strip()}\n"
        f"Normalized Prediction: {str(pred_norm).strip()}\n\n"
        "Each ground-truth item is an alternative acceptable answer. "
        "Judge the prediction as correct when it is clinically equivalent "
        "to any one acceptable answer. Return the raw JSON object directly."
    )


def parse_llm_judge_response(response_text):
    if response_text == "ERROR":
        return {
            "score": "incorrect",
            "reasoning": "API Request Failed.",
        }

    text = response_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {
            "score": "incorrect",
            "reasoning": (
                "JSON Decode Error. "
                f"Raw Output: {text}"
            ),
        }

    if result.get("score") not in {
        "correct",
        "partially_correct",
        "incorrect",
    }:
        return {
            "score": "incorrect",
            "reasoning": (
                "Invalid Score generated. "
                f"Raw Output: {text}"
            ),
        }

    result["reasoning"] = (
        result.get("reasoning")
        or result.get("reason")
        or result.get("explanation")
        or "No reasoning provided by model."
    )

    return result


def safe_accuracy(correct, total):
    return correct / total if total else None


def format_metric(value):
    return "N/A" if value is None else f"{value:.2%}"


def evaluate_checkpoint(
    weights_path,
    loader,
    processor,
    cfg,
    val_loader,
    llm_client,
    llm_cfg,
    biomed_extractor,
):
    """
    完成单个 checkpoint 的完整评测：

    1. 重建并加载模型；
    2. 在 VQA-Med 2021 validation 上本地生成答案；
    3. 记录每条样本的 Router / Gate / Lambda；
    4. 对开放题未严格匹配样本执行 LLM Judge；
    5. 计算 Closed / Open / Overall Accuracy；
    6. 保存逐样本 JSONL、CSV 和 summary.json。
    """
    name = os.path.basename(weights_path)

    # 每个 checkpoint 拥有独立输出目录。
    output_dir = os.path.join(
        cfg.validation_output_dir,
        name,
    )
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print(f"开始评测：{name}")
    print(f"输出目录：{output_dir}")
    print("=" * 60)

    model, base_model = build_model(
        weights_path,
        loader,
        cfg,
        biomed_extractor,
    )

    # A0 没有 Visual Adapter，其余消融读取对应模块。
    adapter = (
        get_adapter(model)
        if cfg.enable_visual_adapter
        else None
    )

    # Qwen3-VL 视觉塔，用于写入和清空 BioMedCLIP 缓存。
    vision_tower = (
        model.base_model
        .model
        .model
        .visual
    )

    # 以视觉塔参数所在设备和 dtype 为准。
    ref_param = next(
        vision_tower.parameters()
    )
    device = ref_param.device
    adapter_dtype = ref_param.dtype

    if cfg.enable_visual_adapter:
        patch_generate_forward(model)

    # 保存所有图文对的预测、标准答案和路由诊断。
    records = []

    # =========================================================
    # Phase 1：本地生成
    # =========================================================
    with torch.no_grad():
        for batch_inputs, metadata in tqdm(
            val_loader,
            desc="Local Inference",
        ):
            # 将 Tensor 输入移动到模型所在设备。
            inputs = {
                key: (
                    value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in batch_inputs.items()
            }

            if cfg.enable_visual_adapter:
                # BioMedCLIP 图像使用模型浮点 dtype；
                # 文本 token 保持整数 dtype。
                image = inputs.pop(
                    "biomed_image_tensors"
                ).to(adapter_dtype)

                text = inputs.pop(
                    "biomed_text_tokens"
                )

                # 每个 batch 只计算一次 BioMedCLIP 图像和文本特征。
                image_feat, text_feat = (
                    model.biomed_extractor(
                        image,
                        text,
                    )
                )

                # 缓存到视觉塔，供 generate() 内部多次 forward 复用。
                vision_tower.current_biomed_img_feat = (
                    image_feat
                )
                vision_tower.current_biomed_txt_feat = (
                    text_feat
                )

            # 贪心解码时只传必要参数。
            generate_args = {
                "max_new_tokens": (
                    cfg.max_new_tokens
                ),
                "do_sample": cfg.do_sample,
            }

            if cfg.do_sample:
                generate_args["temperature"] = (
                    cfg.temperature
                )

            generated = model.generate(
                **inputs,
                **generate_args,
            )

            if cfg.enable_visual_adapter:
                # Fusion 前向会保存当前 batch 的路由结果。
                #
                # routing: [B, 3]，依次对应 F3 / F5 / F7。
                # gate:    [B]，每个图文对一个样本级 Gate。
                # lambda:  标量，全局共享。
                routing = (
                    adapter
                    .latest_routing_weights
                    .detach()
                    .float()
                    .cpu()
                )

                gate = (
                    adapter
                    .latest_soft_gate
                    .detach()
                    .float()
                    .reshape(-1)
                    .cpu()
                )

                lambda_value = float(
                    adapter
                    .latest_lambda
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )

                batch_size = len(metadata)

                # 若维度不符合当前三专家设计，直接终止评测。
                if routing.shape != (
                    batch_size,
                    3,
                ):
                    raise RuntimeError(
                        f"routing shape="
                        f"{tuple(routing.shape)}, "
                        f"expected=({batch_size}, 3)"
                    )

                if gate.shape != (
                    batch_size,
                ):
                    raise RuntimeError(
                        f"gate shape="
                        f"{tuple(gate.shape)}, "
                        f"expected=({batch_size},)"
                    )

                # 当前 batch 使用结束后清空缓存，
                # 防止特征错误复用到下一批样本。
                vision_tower.current_biomed_img_feat = (
                    None
                )
                vision_tower.current_biomed_txt_feat = (
                    None
                )

            else:
                routing = None
                gate = None
                lambda_value = None

            # model.generate() 返回：
            # [输入 prompt token + 新生成 token]。
            # 这里只保留新生成的答案部分。
            generated = [
                output[len(input_ids):]
                for input_ids, output in zip(
                    inputs["input_ids"],
                    generated,
                )
            ]

            predictions = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # 将当前 batch 拆成逐样本记录。
            for i, prediction in enumerate(
                predictions
            ):
                meta = metadata[i]

                references_raw = [
                    str(reference).strip()
                    for reference in meta.get(
                        "references",
                        [],
                    )
                    if str(reference).strip()
                ]

                if not references_raw:
                    gt_answer = str(
                        meta.get(
                            "gt_answer",
                            "",
                        )
                    ).strip()

                    if gt_answer:
                        references_raw = [
                            gt_answer
                        ]

                references_norm = (
                    vqa_med_2021_references_eval_cleaning(
                        references_raw
                    )
                )

                if not references_norm:
                    raise ValueError(
                        "参考答案清洗后为空："
                        f"index={meta.get('index')}"
                    )

                pred_raw = prediction.strip()
                pred_norm = (
                    vqa_med_2021_answer_eval_cleaning(
                        pred_raw
                    )
                )

                matched = (
                    pred_norm in references_norm
                )

                category = (
                    "closed"
                    if (
                        str(
                            meta.get(
                                "answer_type",
                                "",
                            )
                        )
                        .strip()
                        .upper()
                        == "CLOSED"
                    )
                    else "open"
                )

                record = {
                    "index": meta.get("index"),
                    "image_path": meta.get(
                        "image_path"
                    ),
                    "question": meta["question"],
                    "question_category": category,
                    "references_raw": references_raw,
                    "references_norm": references_norm,
                    "pred_raw": pred_raw,
                    "pred_norm": pred_norm,
                    "is_norm_match": matched,
                    "llm_judge_score": None,
                    "llm_judge_reason": None,
                    "llm_judge_raw_response": None,
                    "final_correct": matched,
                    "routing_f3": None,
                    "routing_f5": None,
                    "routing_f7": None,
                    "residual_gate": None,
                    "lambda_value": None,
                    "effective_residual_scale": None,
                }

                if routing is not None:
                    record["routing_f3"] = float(
                        routing[i, 0]
                    )
                    record["routing_f5"] = float(
                        routing[i, 1]
                    )
                    record["routing_f7"] = float(
                        routing[i, 2]
                    )
                    record["residual_gate"] = float(
                        gate[i]
                    )
                    record["lambda_value"] = (
                        lambda_value
                    )
                    record[
                        "effective_residual_scale"
                    ] = (
                        lambda_value
                        * float(gate[i])
                    )

                records.append(record)

    # 本地生成结束后释放大模型显存。
    del model
    del base_model
    torch.cuda.empty_cache()

    # =========================================================
    # Phase 2：LLM Judge
    # =========================================================
    system_prompt = (
        llm_cfg.medical_vqa_llm_judge_system_prompt
        + "\nFor VQA-Med 2021, multiple ground truths are "
          "alternative acceptable answers. A prediction is correct "
          "if it is clinically equivalent to any one of them."
    )

    for record in tqdm(
        records,
        desc="LLM Judging",
    ):
        if (
            record["question_category"]
            == "open"
            and not record["is_norm_match"]
        ):
            prompt = build_llm_judge_user_prompt(
                question=record["question"],
                references_raw=(
                    record["references_raw"]
                ),
                references_norm=(
                    record["references_norm"]
                ),
                pred_raw=record["pred_raw"],
                pred_norm=record["pred_norm"],
            )

            response = llm_client.ask(
                system_prompt,
                prompt,
            )

            judged = parse_llm_judge_response(
                response
            )

            record[
                "llm_judge_raw_response"
            ] = response
            record["llm_judge_score"] = (
                judged["score"]
            )
            record["llm_judge_reason"] = (
                judged["reasoning"]
            )

            record["final_correct"] = (
                judged["score"] == "correct"
            )

    # =========================================================
    # Phase 3：准确率统计
    # =========================================================
    closed = [
        item
        for item in records
        if item["question_category"]
        == "closed"
    ]

    opened = [
        item
        for item in records
        if item["question_category"]
        == "open"
    ]

    closed_correct = sum(
        item["final_correct"]
        for item in closed
    )

    open_correct = sum(
        item["final_correct"]
        for item in opened
    )

    exact_correct = sum(
        item["is_norm_match"]
        for item in records
    )

    result = {
        "checkpoint": name,
        "checkpoint_path": os.path.abspath(
            weights_path
        ),
        "closed_acc": safe_accuracy(
            closed_correct,
            len(closed),
        ),
        "open_strict_acc": safe_accuracy(
            open_correct,
            len(opened),
        ),
        "overall_strict_acc": (
            closed_correct
            + open_correct
        ) / len(records),
        "normalized_exact_any_reference": (
            exact_correct / len(records)
        ),
    }

    # =========================================================
    # Phase 4：Router / Gate / Lambda 诊断
    # =========================================================
    if cfg.enable_visual_adapter:
        # 将逐样本记录重新整理为 Tensor，计算整体统计。
        routing = torch.tensor(
            [
                [
                    item["routing_f3"],
                    item["routing_f5"],
                    item["routing_f7"],
                ]
                for item in records
            ]
        )

        gates = torch.tensor(
            [
                item["residual_gate"]
                for item in records
            ]
        )

        effective = torch.tensor(
            [
                item[
                    "effective_residual_scale"
                ]
                for item in records
            ]
        )

        # H(w) = -sum(w * log(w))。
        # 三专家最大熵为 log(3)。
        safe_routing = routing.clamp_min(
            1e-12
        )
        entropy = -(
            safe_routing
            * safe_routing.log()
        ).sum(dim=-1)

        mean_weights = routing.mean(
            dim=0
        )

        gate_q = torch.quantile(
            gates,
            torch.tensor(
                [0.1, 0.5, 0.9]
            ),
        )

        effective_q = torch.quantile(
            effective,
            torch.tensor(
                [0.1, 0.5, 0.9]
            ),
        )

        # 同时写入 summary.json。
        result["routing"] = {
            "mean_f3": float(
                mean_weights[0]
            ),
            "mean_f5": float(
                mean_weights[1]
            ),
            "mean_f7": float(
                mean_weights[2]
            ),
            "gate_mean": float(
                gates.mean()
            ),
            "gate_std": float(
                gates.std(
                    unbiased=False
                )
            ),
            "lambda": records[0][
                "lambda_value"
            ],
            "effective_mean": float(
                effective.mean()
            ),
            "effective_std": float(
                effective.std(
                    unbiased=False
                )
            ),
            "entropy_mean": float(
                entropy.mean()
            ),
            "entropy_normalized": float(
                entropy.mean()
                / math.log(3)
            ),
        }

        print("\n" + "=" * 70)
        print(
            "三尺度路由诊断 | "
            f"checkpoint={name}"
        )
        print("=" * 70)

        print(
            "平均专家权重："
            f"F3={mean_weights[0]:.6f}, "
            f"F5={mean_weights[1]:.6f}, "
            f"F7={mean_weights[2]:.6f}"
        )

        print(
            "residual gate："
            f"mean={gates.mean():.6f}, "
            f"std="
            f"{gates.std(unbiased=False):.6f}, "
            f"P10={gate_q[0]:.6f}, "
            f"P50={gate_q[1]:.6f}, "
            f"P90={gate_q[2]:.6f}"
        )

        print(
            f"lambda："
            f"{records[0]['lambda_value']:.6f}"
        )

        print(
            "lambda × gate："
            f"mean={effective.mean():.6f}, "
            f"std="
            f"{effective.std(unbiased=False):.6f}, "
            f"P10={effective_q[0]:.6f}, "
            f"P50={effective_q[1]:.6f}, "
            f"P90={effective_q[2]:.6f}"
        )

        print(
            "路由 entropy："
            f"mean={entropy.mean():.6f}, "
            f"归一化="
            f"{entropy.mean() / math.log(3):.6f}"
        )

        print("=" * 70)

    # =========================================================
    # Phase 5：输出逐样本记录和汇总结果
    # =========================================================

    # JSONL：一行一个样本，适合后续 Python 逐行读取。
    jsonl_path = os.path.join(
        output_dir,
        "samples.jsonl",
    )

    with open(
        jsonl_path,
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # CSV：方便 Excel、Pandas 或统计软件查看。
    csv_path = os.path.join(
        output_dir,
        "samples.csv",
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                records[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(records)

    # 当前 checkpoint 的整体指标与路由统计。
    summary_path = os.path.join(
        output_dir,
        "summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"逐样本 JSONL："
        f"{jsonl_path}"
    )
    print(
        f"逐样本 CSV："
        f"{csv_path}"
    )
    print(
        f"汇总 JSON："
        f"{summary_path}"
    )

    return result


def get_checkpoints(cfg):
    final_weights = os.path.join(
        cfg.stage2_run_dir,
        "final_weights",
    )

    checkpoints = glob.glob(
        os.path.join(
            cfg.stage2_run_dir,
            "checkpoint-*",
        )
    )

    checkpoints.sort(
        key=lambda path: int(
            path.rsplit("-", 1)[-1]
        )
    )

    checkpoints.append(
        final_weights
    )

    return checkpoints


def main():
    cfg = Stage2EvalConfig()

    # 创建当前实验统一评测目录。
    os.makedirs(
        cfg.validation_output_dir,
        exist_ok=True,
    )

    checkpoints = get_checkpoints(
        cfg
    )

    print(
        f"发现 {len(checkpoints)} "
        "个评测节点："
    )

    for path in checkpoints:
        print(
            f"  - {os.path.basename(path)}"
        )

    # =========================================================
    # 1. Qwen3-VL Loader 与 Processor
    # =========================================================
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=(
            cfg.bnb_4bit_quant_type
        ),
        bnb_4bit_use_double_quant=(
            cfg.bnb_4bit_use_double_quant
        ),
        bnb_4bit_compute_dtype=(
            cfg.bnb_4bit_compute_dtype
        ),
        torch_dtype=cfg.torch_dtype,
        attn_implementation=(
            cfg.attn_implementation
        ),
        device_map="auto",
    )

    processor = loader.load_processor()

    # 批量自回归生成时使用 left padding。
    processor.tokenizer.padding_side = (
        "left"
    )

    # =========================================================
    # 2. BioMedCLIP
    # =========================================================
    biomed_extractor = None
    biomed_transform = None
    biomed_tokenizer = None

    # A0 不需要 BioMedCLIP，其余启用 Visual Adapter 的实验加载。
    if cfg.enable_visual_adapter:
        (
            biomed_extractor,
            biomed_transform,
            biomed_tokenizer,
        ) = load_biomedclip(
            biomedclip_path=(
                cfg.biomedclip_path
            ),
            print_rank=cfg.print_rank,
        )

    # =========================================================
    # 3. VQA-Med 2021 Validation DataLoader
    # =========================================================
    dataset = VQAMED2021Dataset(
        jsonl_path=(
            cfg.vqa_med_2021_validation_jsonl_path
        ),
        expected_split="validation",
        expected_count=(
            cfg.vqa_med_2021_validation_expected_count
        ),
        verify_images=(
            cfg.vqa_med_2021_verify_images
        ),
        strict=(
            cfg.vqa_med_2021_dataset_strict
        ),
    )

    collator = VQAMED2021EvalCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=(
            biomed_transform
        ),
        biomed_tokenizer=(
            biomed_tokenizer
        ),
    )

    val_loader = DataLoader(
        dataset,
        batch_size=(
            cfg.per_device_eval_batch_size
        ),
        collate_fn=collator,
        num_workers=(
            cfg.dataloader_num_workers
        ),
        shuffle=False,
    )

    # =========================================================
    # 4. LLM Judge
    # =========================================================
    llm_cfg = LLMAPIConfig()

    llm_client = DeepSeekClient.from_config(
        llm_cfg
    )

    print(
        "LLM Judge："
        f"model={llm_cfg.judge_model_name}, "
        "thinking=disabled, "
        f"temperature={llm_cfg.temperature}"
    )

    # =========================================================
    # 5. 逐 checkpoint 评测
    # =========================================================
    results = [
        evaluate_checkpoint(
            path,
            loader,
            processor,
            cfg,
            val_loader,
            llm_client,
            llm_cfg,
            biomed_extractor,
        )
        for path in checkpoints
    ]

    # 保存所有 checkpoint 的汇总排行榜。
    leaderboard_path = os.path.join(
        cfg.validation_output_dir,
        "leaderboard.json",
    )

    with open(
        leaderboard_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # =========================================================
    # 6. 终端输出排行榜
    # =========================================================
    print(
        "\n\n"
        + "🏆" * 20
        + " VQA-Med 2021 评测结果 "
        + "🏆" * 20
    )

    print(
        f"| 评测节点 "
        f"({cfg.ablation_id}) "
        "| Closed Acc "
        "| Open Acc "
        "| Overall Acc "
        "| Exact-Any-Ref |"
    )

    print(
        "| :--- | :---: "
        "| :---: | :---: | :---: |"
    )

    for result in results:
        print(
            f"| {result['checkpoint']} "
            f"| {format_metric(result['closed_acc'])} "
            f"| {format_metric(result['open_strict_acc'])} "
            f"| {format_metric(result['overall_strict_acc'])} "
            f"| {format_metric(result['normalized_exact_any_reference'])} |"
        )

    best = max(
        results,
        key=lambda item: (
            item["overall_strict_acc"]
        ),
    )

    best_checkpoint = {
        "dataset": "VQA-Med 2021",
        "selection_split": "validation",
        "selection_metric": "overall_strict_acc",
        "selected_checkpoint": (
            best["checkpoint"]
        ),
        "selected_checkpoint_path": (
            best["checkpoint_path"]
        ),
        "validation_score": (
            best["overall_strict_acc"]
        ),
        "ablation_id": cfg.ablation_id,
        "seed": cfg.seed,
        "stage1_seed": cfg.stage1_seed,
    }

    with open(
        cfg.best_checkpoint_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_checkpoint,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\n最佳节点："
        f"{best['checkpoint']} "
        f"(Overall: "
        f"{best['overall_strict_acc']:.2%})"
    )

    print(
        f"评测输出目录："
        f"{cfg.validation_output_dir}"
    )

    print(
        f"排行榜："
        f"{leaderboard_path}"
    )


if __name__ == "__main__":
    main()