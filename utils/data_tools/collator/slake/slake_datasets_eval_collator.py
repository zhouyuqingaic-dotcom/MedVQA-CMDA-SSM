from typing import Any, Sequence

import torch
from PIL import Image

from utils.data_tools.prompt_builder.slake_prompt_builder import (
    build_slake_prompt,
)


class SLAKEEvalCollator:
    """
    SLAKE Stage 2 评测 Collator。

    A0：
        只构造 Qwen3-VL 输入。

    A1～A5：
        同时构造 Qwen3-VL 与 BioMedCLIP 输入。
    """

    def __init__(
        self,
        processor: Any,
        cfg: Any,
        biomed_transform: Any = None,
        biomed_tokenizer: Any = None,
    ):
        self.processor = processor
        self.max_size = int(cfg.slake_max_size)
        self.instruction_suffix = cfg.slake_instruction_suffix
        self.enable_visual_adapter = bool(
            cfg.enable_visual_adapter
        )

        if self.processor.tokenizer.padding_side != "left":
            raise ValueError(
                "SLAKEEvalCollator 要求 "
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
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"读取 SLAKE 图像失败：{image_path}"
            ) from exc

    def _resize_for_qwen(
        self,
        image: Image.Image,
    ) -> Image.Image:
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

    def __call__(
        self,
        batch: Sequence[dict[str, Any]],
    ):
        if not batch:
            raise ValueError(
                "SLAKEEvalCollator 收到了空 batch。"
            )

        prompt_texts: list[str] = []
        qwen_images: list[Image.Image] = []
        metadata_list: list[dict[str, Any]] = []

        biomed_images: list[torch.Tensor] = []
        biomed_questions: list[str] = []

        for sample in batch:
            image_path = sample["image_path"]

            raw_question = str(
                sample["question"]
            ).strip()

            if not raw_question:
                raise ValueError(
                    f"发现空问题，图像路径：{image_path}"
                )

            # Qwen 使用附带回答要求的完整问题。
            qwen_question = build_slake_prompt(
                question=raw_question,
                instruction_suffix=self.instruction_suffix,
            )

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
                        "biomed_transform 应返回 Tensor，"
                        f"当前类型为 {type(biomed_image)}。"
                    )

                biomed_images.append(
                    biomed_image
                )

                # 与 Stage 2 训练一致：
                # Router 使用原始医学问题，不附加回答格式指令。
                biomed_questions.append(
                    raw_question
                )

            qwen_image = self._resize_for_qwen(
                original_image
            )

            prompt_texts.append(prompt_text)
            qwen_images.append(qwen_image)

            metadata_list.append(
                {
                    "index": sample.get("index"),
                    "image_path": image_path,
                    "question": raw_question,
                    "gt_answer": sample.get(
                        "answer",
                        "",
                    ),
                    "question_type": sample.get(
                        "question_type",
                        "UNKNOWN",
                    ),
                    "answer_type": sample.get(
                        "answer_type",
                        "UNKNOWN",
                    ),
                }
            )

        batch_inputs = self.processor(
            text=prompt_texts,
            images=qwen_images,
            return_tensors="pt",
            padding=True,
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
                    "biomed_tokenizer 应返回 Tensor，"
                    f"当前类型为 "
                    f"{type(biomed_text_tokens)}。"
                )

            if (
                biomed_text_tokens.ndim != 2
                or biomed_text_tokens.shape[0]
                != len(batch)
            ):
                raise ValueError(
                    "BioMedCLIP 文本 token 形状错误："
                    f"{tuple(biomed_text_tokens.shape)}"
                )

            batch_inputs["biomed_text_tokens"] = (
                biomed_text_tokens
                .detach()
                .cpu()
            )

        return batch_inputs, metadata_list
