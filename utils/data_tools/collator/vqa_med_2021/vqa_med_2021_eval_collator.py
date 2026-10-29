from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from typing import Any, Sequence

import torch
from PIL import Image

from utils.data_tools.prompt_builder.vqa_med_2021_prompt_builder import (
    build_vqa_med_2021_prompt,
)


class VQAMED2021EvalCollator:
    """
    VQA-Med 2021 Task 1 的 Stage 2 Validation/Test Collator。

    A0
    --
    只构造 Qwen3-VL 推理输入。

    启用 Visual Adapter
    -------------------
    同时构造：
        biomed_image_tensors
        biomed_text_tokens

    防止答案泄漏
    ------------
    送入模型的 batch_inputs 只由以下信息构造：
        image_path
        question

    answer 和 references 只写入 metadata_list，供模型完成生成后
    进行离线评测，不会进入 Qwen Prompt 或 BioMedCLIP Router。

    Dataset 输入字段
    ----------------
    每个 sample 应包含：

        {
            "index": int,
            "split": str,
            "image_id": str,
            "image_name": str,
            "image_path": str,
            "question": str,
            "answer": str,
            "references": list[str],
            "question_type": str,
            "answer_type": str,
        }

    配置字段
    --------
    cfg 至少应包含：

        cfg.vqa_med_2021_max_size
        cfg.vqa_med_2021_instruction_suffix
        cfg.enable_visual_adapter
    """

    def __init__(
        self,
        processor: Any,
        cfg: Any,
        biomed_transform: Any = None,
        biomed_tokenizer: Any = None,
    ) -> None:
        self.processor = processor

        if not hasattr(self.processor, "tokenizer"):
            raise AttributeError(
                "VQAMED2021EvalCollator 要求 processor 具有 tokenizer。"
            )

        self.max_size = int(
            cfg.vqa_med_2021_max_size
        )
        self.instruction_suffix = str(
            cfg.vqa_med_2021_instruction_suffix
        )
        self.enable_visual_adapter = bool(
            cfg.enable_visual_adapter
        )

        if self.max_size <= 0:
            raise ValueError(
                "vqa_med_2021_max_size 必须大于 0，"
                f"当前值为 {self.max_size}。"
            )

        if not self.instruction_suffix.strip():
            raise ValueError(
                "vqa_med_2021_instruction_suffix 不能为空。"
            )

        # generate() 批量推理要求左侧 padding，
        # 使不同长度样本的最后一个有效 token 对齐在右侧。
        if self.processor.tokenizer.padding_side != "left":
            raise ValueError(
                "VQAMED2021EvalCollator 要求 "
                "processor.tokenizer.padding_side='left'。"
            )

        self.biomed_img_transform = biomed_transform
        self.biomed_tokenizer = biomed_tokenizer

        if self.enable_visual_adapter:
            if self.biomed_img_transform is None:
                raise ValueError(
                    "enable_visual_adapter=True，"
                    "但没有传入 biomed_transform。"
                )

            if self.biomed_tokenizer is None:
                raise ValueError(
                    "enable_visual_adapter=True，"
                    "但没有传入 biomed_tokenizer。"
                )

    @staticmethod
    def _load_rgb_image(
        image_path: str,
    ) -> Image.Image:
        """
        从磁盘读取图像，并返回与文件句柄解耦的 RGB 图像。
        """
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                "读取 VQA-Med 2021 图像失败："
                f"{image_path}"
            ) from exc

    def _resize_for_qwen(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        等比例限制送入 Qwen3-VL 的图像最长边。
        """
        width, height = image.size
        longest_edge = max(width, height)

        if longest_edge <= self.max_size:
            return image

        scale = self.max_size / longest_edge

        new_width = max(
            1,
            round(width * scale),
        )
        new_height = max(
            1,
            round(height * scale),
        )

        bicubic = getattr(
            Image,
            "Resampling",
            Image,
        ).BICUBIC

        return image.resize(
            (new_width, new_height),
            resample=bicubic,
        )

    @staticmethod
    def _normalize_references(
        sample: dict[str, Any],
        gt_answer: str,
        image_path: str,
    ) -> list[str]:
        """
        读取并保留原始参考答案。

        此处不做评测归一化。大小写、标点和医学表达保持原样，
        后续由 vqa_med_2021_references_eval_cleaning 统一处理。

        若旧样本没有 references，则回退为 [gt_answer]。
        """
        raw_references = sample.get(
            "references"
        )

        if raw_references is None:
            references = (
                [gt_answer]
                if gt_answer
                else []
            )
        elif isinstance(
            raw_references,
            str,
        ):
            reference = raw_references.strip()
            references = (
                [reference]
                if reference
                else []
            )
        elif isinstance(
            raw_references,
            SequenceABC,
        ):
            references = [
                str(reference).strip()
                for reference in raw_references
                if reference is not None
                and str(reference).strip()
            ]
        else:
            raise TypeError(
                "VQA-Med 2021 references 必须是字符串、序列或 None，"
                f"image_path={image_path}, "
                f"当前类型为 {type(raw_references)}。"
            )

        # 按原顺序去重，不做 lower、不改标点。
        references = list(
            dict.fromkeys(references)
        )

        if not references and gt_answer:
            references = [gt_answer]

        if not references:
            raise ValueError(
                "VQA-Med 2021 评测样本没有有效参考答案，"
                f"image_path={image_path}"
            )

        if gt_answer and gt_answer not in references:
            # 当前统一 JSONL 的 answer 应是 references[0]。
            # 为了兼容未来轻微格式变化，不在 Collator 中强制报错，
            # 而是把主答案补到首位，保证评测信息完整。
            references = [
                gt_answer,
                *references,
            ]

        return references

    @staticmethod
    def _validate_processor_batch(
        batch_inputs: Any,
        expected_batch_size: int,
    ) -> None:
        """
        检查 Processor 的推理输出。
        """
        required_keys = {
            "input_ids",
            "attention_mask",
        }

        missing_keys = [
            key
            for key in required_keys
            if key not in batch_inputs
        ]

        if missing_keys:
            raise KeyError(
                "VQA-Med 2021 Eval Processor 输出缺少字段："
                f"{missing_keys}"
            )

        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                "batch_inputs['input_ids'] 应为 torch.Tensor，"
                f"当前类型为 {type(input_ids)}。"
            )

        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(
                "batch_inputs['attention_mask'] 应为 torch.Tensor，"
                f"当前类型为 {type(attention_mask)}。"
            )

        if input_ids.ndim != 2:
            raise ValueError(
                "batch_inputs['input_ids'] 必须为二维张量，"
                f"当前形状为 {tuple(input_ids.shape)}。"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "input_ids 与 attention_mask 形状不一致："
                f"input_ids={tuple(input_ids.shape)}, "
                f"attention_mask={tuple(attention_mask.shape)}"
            )

        if input_ids.shape[0] != expected_batch_size:
            raise ValueError(
                "Eval Processor batch size 错误："
                f"actual={input_ids.shape[0]}, "
                f"expected={expected_batch_size}"
            )

    def __call__(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not batch:
            raise ValueError(
                "VQAMED2021EvalCollator 收到了空 batch。"
            )

        prompt_texts: list[str] = []
        qwen_images: list[Image.Image] = []
        metadata_list: list[dict[str, Any]] = []

        biomed_images: list[torch.Tensor] = []
        biomed_questions: list[str] = []

        for batch_index, sample in enumerate(batch):
            if not isinstance(sample, dict):
                raise TypeError(
                    "VQA-Med 2021 评测 batch 中的每个样本必须是 dict，"
                    f"batch_index={batch_index}, "
                    f"当前类型为 {type(sample)}。"
                )

            image_path = str(
                sample.get("image_path", "")
            ).strip()

            if not image_path:
                raise ValueError(
                    "VQA-Med 2021 评测样本 image_path 为空，"
                    f"batch_index={batch_index}。"
                )

            raw_question = str(
                sample.get("question", "")
            ).strip()

            if not raw_question:
                raise ValueError(
                    "发现空问题，"
                    f"batch_index={batch_index}, "
                    f"image_path={image_path}"
                )

            # Qwen 使用带回答格式要求的完整问题。
            qwen_question = build_vqa_med_2021_prompt(
                question=raw_question,
                instruction_suffix=self.instruction_suffix,
            )

            # 只有 image + question 进入模型 Prompt。
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {
                            "type": "text",
                            "text": qwen_question,
                        },
                    ],
                }
            ]

            prompt_text = (
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            if not isinstance(prompt_text, str):
                raise TypeError(
                    "processor.apply_chat_template 应返回 str，"
                    f"当前类型为 {type(prompt_text)}。"
                )

            original_image = self._load_rgb_image(
                image_path
            )

            if self.enable_visual_adapter:
                biomed_image = (
                    self.biomed_img_transform(
                        original_image
                    )
                )

                if not isinstance(
                    biomed_image,
                    torch.Tensor,
                ):
                    raise TypeError(
                        "biomed_transform 应返回 torch.Tensor，"
                        f"batch_index={batch_index}, "
                        f"当前类型为 {type(biomed_image)}。"
                    )

                biomed_images.append(
                    biomed_image
                )

                # Router 只接收原始医学问题。
                biomed_questions.append(
                    raw_question
                )

            qwen_image = self._resize_for_qwen(
                original_image
            )

            prompt_texts.append(
                prompt_text
            )
            qwen_images.append(
                qwen_image
            )

            # -------------------------------------------------
            # Ground Truth 只进入 metadata，不进入模型输入。
            # -------------------------------------------------
            gt_answer = str(
                sample.get("answer", "")
            ).strip()

            if not gt_answer:
                raise ValueError(
                    "VQA-Med 2021 评测样本 answer 为空，"
                    f"batch_index={batch_index}, "
                    f"image_path={image_path}"
                )

            references = self._normalize_references(
                sample=sample,
                gt_answer=gt_answer,
                image_path=image_path,
            )

            metadata_list.append(
                {
                    "index": sample.get(
                        "index"
                    ),
                    "split": sample.get(
                        "split",
                        "UNKNOWN",
                    ),
                    "image_id": sample.get(
                        "image_id",
                        "",
                    ),
                    "image_name": sample.get(
                        "image_name",
                        "",
                    ),
                    "image_path": image_path,
                    "question": raw_question,
                    "gt_answer": gt_answer,
                    "references": references,
                    "question_type": sample.get(
                        "question_type",
                        "Abnormality",
                    ),
                    "answer_type": sample.get(
                        "answer_type",
                        "UNKNOWN",
                    ),
                }
            )

        # =====================================================
        # 批量构造 Qwen3-VL 推理输入
        # =====================================================
        batch_inputs = self.processor(
            text=prompt_texts,
            images=qwen_images,
            return_tensors="pt",
            padding=True,
        )

        self._validate_processor_batch(
            batch_inputs=batch_inputs,
            expected_batch_size=len(batch),
        )

        # =====================================================
        # 附加 BioMedCLIP 输入
        # =====================================================
        if self.enable_visual_adapter:
            if len(biomed_images) != len(batch):
                raise RuntimeError(
                    "BioMedCLIP 图像数量与 batch size 不一致："
                    f"images={len(biomed_images)}, "
                    f"batch={len(batch)}"
                )

            batch_inputs[
                "biomed_image_tensors"
            ] = torch.stack(
                biomed_images,
                dim=0,
            )

            biomed_text_tokens = (
                self.biomed_tokenizer(
                    biomed_questions
                )
            )

            if not isinstance(
                biomed_text_tokens,
                torch.Tensor,
            ):
                raise TypeError(
                    "biomed_tokenizer 应返回 torch.Tensor，"
                    f"当前类型为 "
                    f"{type(biomed_text_tokens)}。"
                )

            if (
                biomed_text_tokens.ndim != 2
                or biomed_text_tokens.shape[0]
                != len(batch)
            ):
                raise ValueError(
                    "BioMedCLIP 文本 token 形状错误，"
                    f"当前形状为 "
                    f"{tuple(biomed_text_tokens.shape)}，"
                    f"预期 batch size={len(batch)}。"
                )

            batch_inputs[
                "biomed_text_tokens"
            ] = (
                biomed_text_tokens
                .detach()
                .cpu()
            )

        return batch_inputs, metadata_list
