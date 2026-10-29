from typing import Any, Sequence

import torch
from PIL import Image

from utils.data_tools.prompt_builder.vqa_rad_prompt_builder import (
    build_vqa_rad_prompt,
)
from utils.data_tools.prompt_cleaning.vqa_rad_answer_cleaning import (
    vqa_rad_answer_train_cleaning,
)


class VQARADTrainCollator:
    """
    VQA-RAD Stage 2 训练数据整理器。

    主要职责：
    1. 为 Qwen3-VL 构造图像问答对话；
    2. 为 RoMA-Net V2-lite 构造 BioMedCLIP 图像和问题输入；
    3. 只对 assistant 的答案计算语言模型损失；
    4. A0 不生成 BioMedCLIP 输入，A1～A6 生成。
    """

    def __init__(
        self,
        processor: Any,
        cfg: Any,
        biomed_transform: Any = None,
        biomed_tokenizer: Any = None,
    ):
        self.processor = processor
        self.max_size = int(cfg.vqa_rad_max_size)
        self.instruction_suffix = cfg.vqa_rad_instruction_suffix
        self.enable_visual_adapter = bool(cfg.enable_visual_adapter)

        if self.processor.tokenizer.padding_side != "right":
            raise ValueError(
                "VQARADTrainCollator 要求 "
                "processor.tokenizer.padding_side='right'。"
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
    def _load_rgb_image(image_path: str) -> Image.Image:
        """读取图像并转换为独立的 RGB 图像。"""
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"读取 VQA-RAD 图像失败：{image_path}"
            ) from exc

    def _resize_for_qwen(self, image: Image.Image) -> Image.Image:
        """限制送入 Qwen3-VL 的图像最长边。"""
        width, height = image.size
        longest_edge = max(width, height)

        if longest_edge <= self.max_size:
            return image

        scale = self.max_size / longest_edge

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        bicubic = getattr(
            Image,
            "Resampling",
            Image,
        ).BICUBIC

        return image.resize(
            (new_width, new_height),
            resample=bicubic,
        )

    def __call__(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError(
                "VQARADTrainCollator 收到了空 batch。"
            )

        prompt_texts: list[str] = []
        full_texts: list[str] = []
        qwen_images: list[Image.Image] = []

        biomed_images: list[torch.Tensor] = []
        biomed_questions: list[str] = []

        for sample in batch:
            image_path = sample["image_path"]

            # 原始医学问题供 BioMedCLIP router 使用。
            raw_question = str(sample["question"]).strip()

            if not raw_question:
                raise ValueError(
                    f"发现空问题，图像路径：{image_path}"
                )

            # Qwen3-VL 使用带回答格式要求的完整指令。
            qwen_question = build_vqa_rad_prompt(
                question=raw_question,
                instruction_suffix=self.instruction_suffix,
            )

            answer_text = vqa_rad_answer_train_cleaning(
                sample["answer"]
            )

            if not answer_text:
                raise ValueError(
                    "发现清洗后为空的 VQA-RAD 答案，"
                    f"图像路径：{image_path}"
                )

            prompt_messages = [
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

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer_text,
                        }
                    ],
                }
            ]

            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            original_image = self._load_rgb_image(
                image_path
            )

            if self.enable_visual_adapter:
                biomed_image = self.biomed_img_transform(
                    original_image
                )

                if not isinstance(
                    biomed_image,
                    torch.Tensor,
                ):
                    raise TypeError(
                        "biomed_transform 应返回 torch.Tensor，"
                        f"当前类型为 {type(biomed_image)}。"
                    )

                biomed_images.append(
                    biomed_image
                )

                # Router 只看原始医学问题，
                # 不附加 Qwen 的回答格式要求。
                biomed_questions.append(
                    raw_question
                )

            qwen_image = self._resize_for_qwen(
                original_image
            )

            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            qwen_images.append(qwen_image)

        # 批量计算 prompt token，避免逐样本调用 Processor。
        prompt_batch = self.processor(
            text=prompt_texts,
            images=qwen_images,
            return_tensors="pt",
            padding=True,
        )

        prompt_lengths = (
            prompt_batch["attention_mask"]
            .sum(dim=1)
        )

        prompt_input_ids = prompt_batch["input_ids"]
        del prompt_batch

        # 构造包含答案的完整训练 batch。
        batch_inputs = self.processor(
            text=full_texts,
            images=qwen_images,
            return_tensors="pt",
            padding=True,
        )

        # 确认 prompt token 是完整序列的严格前缀。
        for index, prompt_length in enumerate(
            prompt_lengths.tolist()
        ):
            prompt_ids = prompt_input_ids[
                index,
                :prompt_length,
            ]

            full_prefix_ids = batch_inputs[
                "input_ids"
            ][
                index,
                :prompt_length,
            ]

            if not torch.equal(
                prompt_ids,
                full_prefix_ids,
            ):
                raise RuntimeError(
                    "Prompt token 与完整 VQA-RAD 序列前缀不一致，"
                    f"样本索引：{index}"
                )

        if self.enable_visual_adapter:
            batch_inputs["biomed_image_tensors"] = (
                torch.stack(
                    biomed_images,
                    dim=0,
                )
            )

            biomed_text_tokens = self.biomed_tokenizer(
                biomed_questions
            )

            if not isinstance(
                biomed_text_tokens,
                torch.Tensor,
            ):
                raise TypeError(
                    "biomed_tokenizer 应返回 torch.Tensor，"
                    f"当前类型为 {type(biomed_text_tokens)}。"
                )

            if (
                biomed_text_tokens.ndim != 2
                or biomed_text_tokens.shape[0] != len(batch)
            ):
                raise ValueError(
                    "BioMedCLIP 文本 token 形状错误，"
                    f"当前形状为 "
                    f"{tuple(biomed_text_tokens.shape)}，"
                    f"预期 batch size={len(batch)}。"
                )

            batch_inputs["biomed_text_tokens"] = (
                biomed_text_tokens
                .detach()
                .cpu()
            )

        labels = batch_inputs["input_ids"].clone()

        # 屏蔽 user prompt、视觉 token 和 assistant 起始标记。
        for index, prompt_length in enumerate(
            prompt_lengths.tolist()
        ):
            labels[index, :prompt_length] = -100

        # 只屏蔽真正的 padding。
        labels.masked_fill_(
            batch_inputs["attention_mask"].eq(0),
            -100,
        )

        valid_token_counts = labels.ne(-100).sum(
            dim=1
        )

        if torch.any(valid_token_counts == 0):
            invalid_indices = (
                torch.nonzero(
                    valid_token_counts == 0,
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )

            raise RuntimeError(
                "部分 VQA-RAD 样本没有有效答案监督 token，"
                f"样本索引：{invalid_indices}"
            )

        batch_inputs["labels"] = labels
        return batch_inputs