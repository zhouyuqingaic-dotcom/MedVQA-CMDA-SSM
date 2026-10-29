from typing import Any, Sequence

import torch
from PIL import Image

from utils.data_tools.prompt_builder.mimic_cxr_prompt_builder import (
    build_mimic_cxr_prompt,
)
from utils.data_tools.prompt_cleaning.mimic_cxr_text_cleaning import (
    mimic_cxr_text_train_cleaning,
)


class MIMICCXRTrainCollator:
    """
    MIMIC-CXR Stage 1 训练数据整理器。

    主要职责：
    1. 构造 Qwen3-VL 的图像与报告生成对话；
    2. 仅为 A2/A3/A3-NoRMS 构造冻结 BiomedCLIP 图文先验；
    3. 只对 assistant 的目标报告计算语言模型损失；
    4. 屏蔽用户指令、视觉 token 和 padding token。

    设计原则：
    - A0 是 LoRA-only；A1 是 Uniform Fusion，均不使用 BioMedCLIP；
    - A2/A3/A3-NoRMS 使用固定报告指令作为条件文本；
    - 不把真实 IMPRESSION 或语言模型 labels 放进 BiomedCLIP；
    - 是否提取先验以 cfg.requires_biomedclip 为准。
    """

    def __init__(
        self,
        processor: Any,
        cfg: Any,
        biomed_transform: Any = None,
        biomed_tokenizer: Any = None,
    ):
        self.processor = processor

        # 新版代码直接读取明确配置，字段缺失时立即报错。
        self.max_size = int(cfg.mimic_cxr_max_size)
        self.enable_visual_adapter = bool(cfg.enable_visual_adapter)

        # 不能仅判断 enable_visual_adapter。
        # A1 虽然有四方向 SSM，但使用 Uniform Fusion，
        # 完全没有 Query/Key 和 BiomedCLIP 条件分支。
        self.requires_biomedclip = bool(cfg.requires_biomedclip)

        if self.requires_biomedclip and not self.enable_visual_adapter:
            raise ValueError(
                "requires_biomedclip=True 与关闭视觉 Adapter 冲突。"
            )

        if cfg.biomed_condition_text_mode != "fixed_instruction":
            raise ValueError(
                "MIMIC-CXR Stage 1 的跨模态文本必须是固定报告指令，"
                "不能使用真实 IMPRESSION。"
            )

        # MIMIC-CXR 使用固定报告生成指令，只需构造一次。
        self.question_text = build_mimic_cxr_prompt(
            instruction=cfg.mimic_cxr_instruction_suffix
        )

        # Stage 1 trainer 已设置右侧 padding。
        # 在这里显式检查，避免标签边界计算出现静默错误。
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError(
                "MIMICCXRTrainCollator 要求 "
                "processor.tokenizer.padding_side='right'。"
            )

        self.biomed_img_transform = biomed_transform
        self.biomed_tokenizer = biomed_tokenizer
        self.biomed_text_token = None

        if self.requires_biomedclip:
            if self.biomed_img_transform is None:
                raise ValueError(
                    "当前消融需要 BiomedCLIP，但没有传入 biomed_transform。"
                )

            if self.biomed_tokenizer is None:
                raise ValueError(
                    "当前消融需要 BiomedCLIP，但没有传入 biomed_tokenizer。"
                )

            # MIMIC-CXR 的 router 文本是固定指令，
            # 因此只需要执行一次 BioMedCLIP 文本分词。
            tokenized_text = self.biomed_tokenizer([self.question_text])

            if not isinstance(tokenized_text, torch.Tensor):
                raise TypeError(
                    "biomed_tokenizer 应返回 torch.Tensor，"
                    f"当前类型为 {type(tokenized_text)}。"
                )

            if tokenized_text.ndim != 2 or tokenized_text.shape[0] != 1:
                raise ValueError(
                    "BioMedCLIP 文本 token 的预期形状为 [1, context_length]，"
                    f"当前形状为 {tuple(tokenized_text.shape)}。"
                )

            # Collator 输出保持在 CPU，之后由 Trainer 自动移动到目标设备。
            self.biomed_text_token = tokenized_text[0].detach().cpu()

    @staticmethod
    def _load_rgb_image(image_path: str) -> Image.Image:
        """
        从磁盘读取图像，并转换为 RGB。

        这里单独复制为新的 RGB 图像，因此退出 with 语句后，
        返回的图像仍然可以安全使用。
        """
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"读取 MIMIC-CXR 图像失败：{image_path}"
            ) from exc

    def _resize_for_qwen(self, image: Image.Image) -> Image.Image:
        """
        限制输入 Qwen3-VL 的最长边。

        BioMedCLIP 使用它自己的 transform，因此这里只调整
        送入 Qwen3-VL Processor 的图像。
        """
        width, height = image.size
        longest_edge = max(width, height)

        if longest_edge <= self.max_size:
            return image

        scale = self.max_size / longest_edge

        # 至少保留一个像素，避免极端尺寸导致宽或高变为 0。
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        # 同时兼容新旧版本 Pillow。
        bicubic = getattr(Image, "Resampling", Image).BICUBIC

        return image.resize(
            (new_width, new_height),
            resample=bicubic,
        )

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("MIMICCXRTrainCollator 收到了空 batch。")

        prompt_texts: list[str] = []
        full_texts: list[str] = []
        qwen_images: list[Image.Image] = []
        biomed_images: list[torch.Tensor] = []

        for sample in batch:
            image_path = sample["image_path"]

            # 清洗目标放射学报告。
            answer_text = mimic_cxr_text_train_cleaning(
                sample.get("target_text", "")
            )

            # 清洗后为空说明该样本没有有效监督信号。
            # 不建议在 DDP Collator 中静默丢弃，否则不同 rank
            # 可能出现不同的有效 batch size。
            if not answer_text:
                raise ValueError(
                    "发现清洗后为空的 MIMIC-CXR 目标报告。"
                    f"图像路径：{image_path}"
                )

            # 用户消息：图像 + 固定报告生成指令。
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
                            "text": self.question_text,
                        },
                    ],
                }
            ]

            # Prompt-only 文本用于确定 assistant 回答的起始位置。
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 完整训练文本包含 assistant 的目标报告。
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

            original_image = self._load_rgb_image(image_path)

            # CM-DA-SSM 的 BioMedCLIP 图像输入使用原始 RGB 图像，
            # resize、crop、normalize 由 BioMedCLIP transform 自己完成。
            if self.requires_biomedclip:
                biomed_image = self.biomed_img_transform(original_image)

                if not isinstance(biomed_image, torch.Tensor):
                    raise TypeError(
                        "biomed_transform 应返回 torch.Tensor，"
                        f"当前类型为 {type(biomed_image)}。"
                    )

                biomed_images.append(biomed_image)

            # Qwen3-VL 图像单独限制最长边。
            qwen_image = self._resize_for_qwen(original_image)

            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            qwen_images.append(qwen_image)

        # ---------------------------------------------------------
        # 一次性批量计算 prompt 长度
        # ---------------------------------------------------------
        # 旧版是在循环中逐个调用 Processor，调用次数为 batch_size + 1。
        # 新版每个 batch 只调用两次 Processor。
        prompt_batch = self.processor(
            text=prompt_texts,
            images=qwen_images,
            return_tensors="pt",
            padding=True,
        )

        # 使用 attention_mask 统计每个 prompt 的有效 token 数量。
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)

        # 只保留用于前缀一致性检查的小张量，
        # 及时释放 prompt 图像预处理产生的大张量。
        prompt_input_ids = prompt_batch["input_ids"]
        del prompt_batch

        # ---------------------------------------------------------
        # 构造完整训练 batch
        # ---------------------------------------------------------
        batch_inputs = self.processor(
            text=full_texts,
            images=qwen_images,
            return_tensors="pt",
            padding=True,
        )

        # ---------------------------------------------------------
        # 检查 prompt 是否确实是完整序列的 token 前缀
        # ---------------------------------------------------------
        # 如果 chat template 或 processor 行为发生变化，
        # 该检查可以防止 labels 被错误地从中间截断。
        for index, prompt_length in enumerate(prompt_lengths.tolist()):
            prompt_ids = prompt_input_ids[index, :prompt_length]
            full_prefix_ids = batch_inputs["input_ids"][index, :prompt_length]

            if not torch.equal(prompt_ids, full_prefix_ids):
                raise RuntimeError(
                    "Prompt token 与完整训练序列前缀不一致，"
                    "无法安全构造 labels。"
                    f"样本索引：{index}"
                )

        # ---------------------------------------------------------
        # 添加 BioMedCLIP 输入
        # ---------------------------------------------------------
        if self.requires_biomedclip:
            batch_size = len(batch)

            batch_inputs["biomed_image_tensors"] = torch.stack(
                biomed_images,
                dim=0,
            )

            # 所有 MIMIC-CXR 样本使用相同的固定指令，
            # 因此直接扩展预先 tokenize 的文本 token。
            batch_inputs["biomed_text_tokens"] = (
                self.biomed_text_token
                .unsqueeze(0)
                .expand(batch_size, -1)
                .clone()
            )

        # ---------------------------------------------------------
        # 构造语言模型监督标签
        # ---------------------------------------------------------
        labels = batch_inputs["input_ids"].clone()

        # 屏蔽 user prompt、图像 token 和 assistant 角色头部。
        # add_generation_prompt=True 生成的 prompt 长度已经包含
        # assistant 回答之前的全部 token。
        for index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[index, :prompt_length] = -100

        # 只通过 attention_mask 屏蔽真正的 padding。
        # 不按 pad_token_id 屏蔽，避免 pad_token_id 与 eos_token_id
        # 相同时误伤正常回答末尾的 EOS。
        labels.masked_fill_(
            batch_inputs["attention_mask"].eq(0),
            -100,
        )

        # 每个样本都应至少包含一个有效监督 token。
        valid_token_counts = labels.ne(-100).sum(dim=1)

        if torch.any(valid_token_counts == 0):
            invalid_indices = (
                torch.nonzero(valid_token_counts == 0, as_tuple=False)
                .flatten()
                .tolist()
            )
            raise RuntimeError(
                "部分样本没有任何有效监督 token，"
                f"样本索引为：{invalid_indices}"
            )

        batch_inputs["labels"] = labels
        return batch_inputs