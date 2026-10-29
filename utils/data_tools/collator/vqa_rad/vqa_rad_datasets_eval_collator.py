from typing import Any, Sequence

import torch
from PIL import Image

from utils.data_tools.prompt_builder.vqa_rad_prompt_builder import (
    build_vqa_rad_prompt,
)


class VQARADEvalCollator:
    """
    VQA-RAD Stage 2 评测/推理 Collator。

    A0：
        只构造 Qwen3-VL 推理输入。

    A1～A6：
        同时构造 Qwen3-VL 与 BioMedCLIP 输入。

    返回：
        batch_inputs:
            Qwen3-VL 推理输入；
            启用视觉 Adapter 时还包含：
                biomed_image_tensors
                biomed_text_tokens

        metadata_list:
            保留样本索引、问题、参考答案和问题类型，
            供评测脚本解码、清洗和计算指标。
    """

    def __init__(
        self,
        processor: Any,
        cfg: Any,
        biomed_transform: Any = None,
        biomed_tokenizer: Any = None,
    ):
        self.processor = processor

        self.max_size = int(
            cfg.vqa_rad_max_size
        )

        self.instruction_suffix = (
            cfg.vqa_rad_instruction_suffix
        )

        self.enable_visual_adapter = bool(
            cfg.enable_visual_adapter
        )

        # 自回归批量生成必须使用左 padding。
        if self.processor.tokenizer.padding_side != "left":
            raise ValueError(
                "VQARADEvalCollator 要求 "
                "processor.tokenizer.padding_side='left'。"
            )

        self.biomed_img_transform = (
            biomed_transform
        )
        self.biomed_tokenizer = (
            biomed_tokenizer
        )

        # A1～A6 必须提供 BioMedCLIP 工具。
        # A0 可以全部为 None。
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
        读取图像并转换为独立的 RGB PIL Image。

        使用 convert 后返回副本，避免离开 with 块后
        原图文件句柄关闭导致延迟读取失败。
        """
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"读取 VQA-RAD 图像失败：{image_path}"
            ) from exc

    def _resize_for_qwen(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        仅限制最长边，不放大原图，并保持宽高比例。
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

        # 同时兼容新旧 Pillow。
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
                "VQARADEvalCollator 收到了空 batch。"
            )

        # Qwen3-VL 输入。
        prompt_texts: list[str] = []
        qwen_images: list[Image.Image] = []

        # 评测信息，不送入模型。
        metadata_list: list[dict[str, Any]] = []

        # A1～A6 的 BioMedCLIP 输入。
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

            # Qwen 使用附带回答格式要求的完整问题。
            qwen_question = build_vqa_rad_prompt(
                question=raw_question,
                instruction_suffix=(
                    self.instruction_suffix
                ),
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

            # 加入 assistant generation prefix，
            # 让 generate() 从 assistant 回答位置继续生成。
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
                # BioMedCLIP 必须接收原始 RGB 图像，
                # 不接收经过 Qwen max_size 缩放后的图像。
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
                        "biomed_transform 应返回 Tensor，"
                        f"当前类型为 "
                        f"{type(biomed_image)}。"
                    )

                biomed_images.append(
                    biomed_image
                )

                # Router 只使用原始医学问题，
                # 不加入 instruction suffix。
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

        # 整个 batch 一次性处理 Qwen3-VL 输入。
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

            # 整个 batch 一次性进行 BioMedCLIP tokenize，
            # 不再循环中逐条 tokenizer。
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
                    f"{tuple(biomed_text_tokens.shape)}，"
                    f"期望 batch size={len(batch)}。"
                )

            batch_inputs["biomed_text_tokens"] = (
                biomed_text_tokens
                .detach()
                .cpu()
            )

        return batch_inputs, metadata_list
