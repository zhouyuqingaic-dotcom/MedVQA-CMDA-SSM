from __future__ import annotations

from typing import Any, Sequence

import torch
from PIL import Image

from utils.data_tools.prompt_builder.vqa_med_2021_prompt_builder import (
    build_vqa_med_2021_prompt,
)
from utils.data_tools.prompt_cleaning.vqa_med_2021_answer_cleaning import (
    vqa_med_2021_answer_train_cleaning,
)


class VQAMED2021TrainCollator:
    """
    VQA-Med 2021 Task 1 的 Stage 2 训练 Collator。

    主要职责
    --------
    1. 为 Qwen3-VL 构造：
       image + question + assistant answer；
    2. 使用 VQA-Med 2021 专属 Prompt Builder；
    3. 使用 VQA-Med 2021 专属训练答案清洗；
    4. 只对 assistant 答案部分计算语言模型损失；
    5. A0 仅构造 Qwen3-VL 输入；
    6. 启用 Visual Adapter 时，同时构造 BioMedCLIP 图像和问题输入。

    Dataset 输入字段
    ----------------
    每个 sample 至少应包含：

        {
            "image_path": str,
            "question": str,
            "answer": str,
        }

    其中 references 不参与训练。训练监督目标只使用 sample["answer"]。

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
                "VQAMED2021TrainCollator 要求 processor 具有 tokenizer。"
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

        # 训练阶段必须右侧 padding。
        #
        # 这样 prompt token 位于序列左侧连续前缀，
        # 可以通过 labels[:, :prompt_length] = -100
        # 准确屏蔽用户问题、视觉 token 和 assistant 起始标记。
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError(
                "VQAMED2021TrainCollator 要求 "
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
        等比例限制 Qwen3-VL 输入图像的最长边。

        小于等于 max_size 的图像不放大；大图使用 bicubic 缩小。
        BioMedCLIP 使用缩放前的 original_image，并由其专属
        transform 负责尺寸和归一化。
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
    def _require_nonempty_text(
        sample: dict[str, Any],
        key: str,
        image_path: str,
    ) -> str:
        """
        从 sample 中读取非空字符串字段。
        """
        if key not in sample:
            raise KeyError(
                "VQA-Med 2021 训练样本缺少字段 "
                f"{key!r}，图像路径：{image_path}"
            )

        value = str(
            sample[key]
        ).strip()

        if not value:
            raise ValueError(
                "VQA-Med 2021 训练样本字段为空："
                f"key={key!r}, image_path={image_path}"
            )

        return value

    @staticmethod
    def _validate_processor_batch(
        batch_inputs: Any,
        expected_batch_size: int,
        batch_name: str,
    ) -> None:
        """
        检查 Processor 是否返回训练所需的基础张量。
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
                f"{batch_name} 缺少 Processor 输出字段："
                f"{missing_keys}"
            )

        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"{batch_name}['input_ids'] 应为 torch.Tensor，"
                f"当前类型为 {type(input_ids)}。"
            )

        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(
                f"{batch_name}['attention_mask'] 应为 torch.Tensor，"
                f"当前类型为 {type(attention_mask)}。"
            )

        if input_ids.ndim != 2:
            raise ValueError(
                f"{batch_name}['input_ids'] 必须为二维张量，"
                f"当前形状为 {tuple(input_ids.shape)}。"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"{batch_name} 的 input_ids 与 attention_mask "
                "形状不一致："
                f"input_ids={tuple(input_ids.shape)}, "
                f"attention_mask={tuple(attention_mask.shape)}"
            )

        if input_ids.shape[0] != expected_batch_size:
            raise ValueError(
                f"{batch_name} batch size 错误："
                f"actual={input_ids.shape[0]}, "
                f"expected={expected_batch_size}"
            )

    def __call__(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        if not batch:
            raise ValueError(
                "VQAMED2021TrainCollator 收到了空 batch。"
            )

        prompt_texts: list[str] = []
        full_texts: list[str] = []
        qwen_images: list[Image.Image] = []

        biomed_images: list[torch.Tensor] = []
        biomed_questions: list[str] = []

        for batch_index, sample in enumerate(batch):
            if not isinstance(sample, dict):
                raise TypeError(
                    "VQA-Med 2021 训练 batch 中的每个样本必须是 dict，"
                    f"batch_index={batch_index}, "
                    f"当前类型为 {type(sample)}。"
                )

            image_path = str(
                sample.get("image_path", "")
            ).strip()

            if not image_path:
                raise ValueError(
                    "VQA-Med 2021 训练样本 image_path 为空，"
                    f"batch_index={batch_index}。"
                )

            # BioMedCLIP Router 使用原始医学问题。
            raw_question = self._require_nonempty_text(
                sample=sample,
                key="question",
                image_path=image_path,
            )

            # Qwen3-VL 使用“原始问题 + 回答格式指令”。
            qwen_question = build_vqa_med_2021_prompt(
                question=raw_question,
                instruction_suffix=self.instruction_suffix,
            )

            # 训练监督只使用主答案 answer。
            #
            # Test 中的多参考答案 references 不参与训练，
            # 也不会被拼入 Prompt。
            raw_answer = self._require_nonempty_text(
                sample=sample,
                key="answer",
                image_path=image_path,
            )

            answer_text = (
                vqa_med_2021_answer_train_cleaning(
                    raw_answer
                )
            )

            if not answer_text:
                raise ValueError(
                    "发现清洗后为空的 VQA-Med 2021 答案，"
                    f"batch_index={batch_index}, "
                    f"image_path={image_path}"
                )

            # -------------------------------------------------
            # 1. 只含用户输入的 Prompt
            # -------------------------------------------------
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

            prompt_text = (
                self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            if not isinstance(prompt_text, str):
                raise TypeError(
                    "processor.apply_chat_template 应返回 str，"
                    f"当前类型为 {type(prompt_text)}。"
                )

            # -------------------------------------------------
            # 2. 用户输入 + assistant 标准答案
            # -------------------------------------------------
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

            full_text = (
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

            if not isinstance(full_text, str):
                raise TypeError(
                    "processor.apply_chat_template 应返回 str，"
                    f"当前类型为 {type(full_text)}。"
                )

            original_image = self._load_rgb_image(
                image_path
            )

            # -------------------------------------------------
            # 3. Visual Adapter / BioMedCLIP 输入
            # -------------------------------------------------
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

                # Router 只看原始问题，不使用带输出要求的 qwen_question。
                biomed_questions.append(
                    raw_question
                )

            # -------------------------------------------------
            # 4. Qwen3-VL 图像输入
            # -------------------------------------------------
            qwen_image = self._resize_for_qwen(
                original_image
            )

            prompt_texts.append(
                prompt_text
            )
            full_texts.append(
                full_text
            )
            qwen_images.append(
                qwen_image
            )

        # =====================================================
        # 5. 单独 tokenize Prompt，用于确定监督起点
        # =====================================================
        prompt_batch = self.processor(
            text=prompt_texts,
            images=qwen_images,
            return_tensors="pt",
            padding=True,
        )

        self._validate_processor_batch(
            batch_inputs=prompt_batch,
            expected_batch_size=len(batch),
            batch_name="prompt_batch",
        )

        prompt_lengths = (
            prompt_batch["attention_mask"]
            .sum(dim=1)
        )

        prompt_input_ids = (
            prompt_batch["input_ids"]
        )
        del prompt_batch

        # =====================================================
        # 6. Tokenize 完整训练序列
        # =====================================================
        batch_inputs = self.processor(
            text=full_texts,
            images=qwen_images,
            return_tensors="pt",
            padding=True,
        )

        self._validate_processor_batch(
            batch_inputs=batch_inputs,
            expected_batch_size=len(batch),
            batch_name="batch_inputs",
        )

        # Prompt 必须是完整序列的严格左侧前缀。
        #
        # 如果 Chat Template、Processor 或 Padding 配置变化，
        # 此检查会立即报错，防止 labels 屏蔽位置悄悄错位。
        for index, prompt_length in enumerate(
            prompt_lengths.tolist()
        ):
            prompt_length = int(prompt_length)

            if prompt_length <= 0:
                raise RuntimeError(
                    "发现长度为 0 的 VQA-Med 2021 Prompt，"
                    f"batch_index={index}。"
                )

            if prompt_length > batch_inputs["input_ids"].shape[1]:
                raise RuntimeError(
                    "Prompt 长度超过完整序列长度："
                    f"batch_index={index}, "
                    f"prompt_length={prompt_length}, "
                    f"full_length={batch_inputs['input_ids'].shape[1]}"
                )

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
                    "Prompt token 与完整 VQA-Med 2021 "
                    "序列前缀不一致，"
                    f"batch_index={index}。"
                )

        # =====================================================
        # 7. 附加 BioMedCLIP 输入
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

            # 保持与 SLAKE / VQA-Med 2019 当前实现一致：
            # Collator 侧先保存在 CPU，Trainer 再统一搬运到设备。
            batch_inputs[
                "biomed_text_tokens"
            ] = (
                biomed_text_tokens
                .detach()
                .cpu()
            )

        # =====================================================
        # 8. 构造 labels：只监督 assistant 答案
        # =====================================================
        labels = (
            batch_inputs["input_ids"]
            .clone()
        )

        # 屏蔽：
        # - system/user 部分；
        # - 视觉 token；
        # - assistant 起始标记。
        for index, prompt_length in enumerate(
            prompt_lengths.tolist()
        ):
            labels[
                index,
                :int(prompt_length),
            ] = -100

        # 屏蔽真正的 padding token。
        #
        # 不直接按 pad_token_id 屏蔽，是因为 pad_token_id 可能与
        # eos_token_id 等特殊 token 共用；attention_mask 更可靠。
        labels.masked_fill_(
            batch_inputs[
                "attention_mask"
            ].eq(0),
            -100,
        )

        valid_token_counts = (
            labels.ne(-100)
            .sum(dim=1)
        )

        if torch.any(
            valid_token_counts == 0
        ):
            invalid_indices = (
                torch.nonzero(
                    valid_token_counts == 0,
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )

            raise RuntimeError(
                "部分 VQA-Med 2021 样本没有有效答案监督 token，"
                f"batch 内索引：{invalid_indices}"
            )

        batch_inputs["labels"] = labels
        return batch_inputs
