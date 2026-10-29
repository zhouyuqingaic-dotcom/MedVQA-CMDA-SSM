#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VQA-Med 2021 Train/Eval Collator 单元测试。

测试目标
--------
1. 不加载 Qwen3-VL 和 BioMedCLIP 大模型，使用轻量 FakeProcessor 测试；
2. 覆盖 A0（LoRA-only）与 A1～A6（Visual Adapter）两条路径；
3. 检查训练 labels 只监督 assistant 答案；
4. 检查 Eval Prompt 不包含 answer/references，避免答案泄漏；
5. 检查 Test 多参考答案完整进入 metadata；
6. 检查 BioMedCLIP Router 只读取原始 question；
7. 检查 Qwen 图像最长边限制；
8. 检查 Train=right padding、Eval=left padding；
9. 检查空 batch 与缺少 BioMedCLIP 依赖时能及时报错。

推荐放置位置
------------
MoRA_Med/other/test_vqa_med_2021_collator.py

运行
----
cd /home/yuqing/RemoteProjects/MoRA_Med

/home/yuqing/miniconda3/envs/qwen3vl/bin/python \
    other/test_vqa_med_2021_collator.py -v
"""

from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from PIL import Image


def _find_project_root() -> Path:
    """
    从脚本位置和当前工作目录向上查找 MoRA_Med 项目根目录。

    项目根目录必须同时包含：
        datas/
        utils/
        config/
    """
    candidates: list[Path] = []

    try:
        candidates.extend(Path(__file__).resolve().parents)
    except NameError:
        pass

    candidates.extend(Path.cwd().resolve().parents)
    candidates.append(Path.cwd().resolve())

    visited: set[Path] = set()

    for candidate in candidates:
        if candidate in visited:
            continue
        visited.add(candidate)

        if (
            (candidate / "datas").is_dir()
            and (candidate / "utils").is_dir()
            and (candidate / "config").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "无法定位 MoRA_Med 项目根目录。请从项目根目录运行：\n"
        "  python other/test_vqa_med_2021_collator.py -v"
    )


PROJECT_ROOT = _find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from datas.vqa_med_2021_datasets import VQAMED2021Dataset

    from utils.data_tools.collator.vqa_med_2021.vqa_med_2021_train_collator import (
        VQAMED2021TrainCollator,
    )
    from utils.data_tools.collator.vqa_med_2021.vqa_med_2021_eval_collator import (
        VQAMED2021EvalCollator,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "VQA-Med 2021 Collator 模块尚未找到。\n"
        "请确认以下文件已经存在：\n"
        "  utils/data_tools/collator/vqa_med_2021/"
        "vqa_med_2021_train_collator.py\n"
        "  utils/data_tools/collator/vqa_med_2021/"
        "vqa_med_2021_eval_collator.py\n"
        "并确认类名分别为：\n"
        "  VQAMED2021TrainCollator\n"
        "  VQAMED2021EvalCollator"
    ) from exc


class FakeTokenizer:
    """只提供 Collator 会读取的 tokenizer 属性。"""

    def __init__(self, padding_side: str) -> None:
        if padding_side not in {"left", "right"}:
            raise ValueError(
                f"padding_side 必须为 left/right，当前为 {padding_side!r}"
            )

        self.padding_side = padding_side
        self.pad_token_id = 0


class FakeProcessor:
    """
    Qwen3-VL Processor 的轻量替身。

    特点
    ----
    - apply_chat_template 生成确定性纯文本；
    - __call__ 完成简单 tokenization 与 padding；
    - 记录每次调用的文本和图像尺寸，便于断言；
    - 不加载任何模型或真实 tokenizer。
    """

    def __init__(self, padding_side: str) -> None:
        self.tokenizer = FakeTokenizer(
            padding_side=padding_side
        )

        self._vocab: dict[str, int] = {
            "<pad>": 0,
        }
        self.call_records: list[dict[str, Any]] = []
        self.chat_template_records: list[dict[str, Any]] = []

    @staticmethod
    def _extract_text(content: Sequence[dict[str, Any]]) -> str:
        text_parts: list[str] = []

        for item in content:
            if item.get("type") == "text":
                text_parts.append(
                    str(item.get("text", ""))
                )

        return " ".join(text_parts).strip()

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        if tokenize:
            raise AssertionError(
                "本测试预期 Collator 调用 "
                "apply_chat_template(tokenize=False)。"
            )

        pieces: list[str] = []

        for message in messages:
            role = str(message.get("role", "")).strip()
            content = message.get("content", [])

            if role == "user":
                pieces.append("<user>")
                pieces.append("<image>")
                pieces.append(
                    self._extract_text(content)
                )
            elif role == "assistant":
                pieces.append("<assistant>")
                pieces.append(
                    self._extract_text(content)
                )
            else:
                raise AssertionError(
                    f"测试 FakeProcessor 收到未知 role={role!r}"
                )

        if add_generation_prompt:
            pieces.append("<assistant>")

        rendered = " ".join(
            piece for piece in pieces if piece
        )

        self.chat_template_records.append(
            {
                "messages": copy.deepcopy(
                    list(messages)
                ),
                "add_generation_prompt": (
                    add_generation_prompt
                ),
                "rendered": rendered,
            }
        )

        return rendered

    def _token_id(self, token: str) -> int:
        token_id = self._vocab.get(token)

        if token_id is None:
            token_id = len(self._vocab)
            self._vocab[token] = token_id

        return token_id

    def _encode(self, text: str) -> list[int]:
        tokens = re.findall(
            r"<[^>]+>|\S+",
            text,
        )
        return [
            self._token_id(token)
            for token in tokens
        ]

    def __call__(
        self,
        *,
        text: Sequence[str],
        images: Sequence[Image.Image],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise AssertionError(
                "本测试预期 return_tensors='pt'。"
            )

        if not padding:
            raise AssertionError(
                "本测试预期 padding=True。"
            )

        texts = list(text)
        image_list = list(images)

        if len(texts) != len(image_list):
            raise AssertionError(
                "FakeProcessor 收到的 text/images 数量不一致。"
            )

        sequences = [
            self._encode(item)
            for item in texts
        ]

        max_length = max(
            len(sequence)
            for sequence in sequences
        )

        input_ids = torch.full(
            (len(sequences), max_length),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(sequences), max_length),
            dtype=torch.long,
        )

        for row_index, sequence in enumerate(
            sequences
        ):
            sequence_tensor = torch.tensor(
                sequence,
                dtype=torch.long,
            )

            if self.tokenizer.padding_side == "right":
                start = 0
            else:
                start = max_length - len(sequence)

            stop = start + len(sequence)

            input_ids[
                row_index,
                start:stop,
            ] = sequence_tensor
            attention_mask[
                row_index,
                start:stop,
            ] = 1

        # Collator 只需要 Processor 返回一个 dict-like batch；
        # 测试不关心真实 Qwen 像素编码，因此给固定形状占位 Tensor。
        pixel_values = torch.zeros(
            (len(image_list), 3, 4, 4),
            dtype=torch.float32,
        )

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        self.call_records.append(
            {
                "texts": list(texts),
                "image_sizes": [
                    image.size
                    for image in image_list
                ],
                "input_ids": input_ids.clone(),
                "attention_mask": (
                    attention_mask.clone()
                ),
            }
        )

        return output


class FakeBioMedTransform:
    """记录输入图像，并返回固定形状 Tensor。"""

    def __init__(self) -> None:
        self.image_sizes: list[tuple[int, int]] = []

    def __call__(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        self.image_sizes.append(
            image.size
        )

        width, height = image.size
        fill_value = float(
            (width + height) % 17
        )

        return torch.full(
            (3, 8, 8),
            fill_value=fill_value,
            dtype=torch.float32,
        )


class FakeBioMedTokenizer:
    """
    记录 Router 文本，并返回 [batch, context_length] Tensor。
    """

    def __init__(
        self,
        context_length: int = 8,
    ) -> None:
        self.context_length = context_length
        self.calls: list[list[str]] = []

    def __call__(
        self,
        questions: Sequence[str],
    ) -> torch.Tensor:
        question_list = [
            str(question)
            for question in questions
        ]
        self.calls.append(
            question_list
        )

        output = torch.zeros(
            (
                len(question_list),
                self.context_length,
            ),
            dtype=torch.long,
        )

        for index, question in enumerate(
            question_list
        ):
            output[index, 0] = len(question)

        return output


def make_cfg(
    *,
    enable_visual_adapter: bool,
) -> SimpleNamespace:
    """
    只构造 Collator 实际需要的最小配置。
    """
    return SimpleNamespace(
        vqa_med_2021_max_size=64,
        vqa_med_2021_instruction_suffix=(
            "QWEN_ONLY_INSTRUCTION: answer briefly."
        ),
        enable_visual_adapter=(
            enable_visual_adapter
        ),
    )


class TestVQAMED2021Collators(
    unittest.TestCase
):
    def setUp(self) -> None:
        self._temp_dir = (
            tempfile.TemporaryDirectory()
        )
        self.temp_root = Path(
            self._temp_dir.name
        )

        self.image_dir = (
            self.temp_root / "images"
        )
        self.image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 第一张图超过 max_size=64，
        # 应被 Qwen 路径缩放到 64×32。
        self.large_image_path = (
            self.image_dir / "synpic_large.jpg"
        )
        Image.new(
            "RGB",
            (160, 80),
            color=(255, 0, 0),
        ).save(
            self.large_image_path
        )

        # 第二张图无需缩放。
        self.small_image_path = (
            self.image_dir / "synpic_small.jpg"
        )
        Image.new(
            "RGB",
            (40, 50),
            color=(0, 255, 0),
        ).save(
            self.small_image_path
        )

        self.train_jsonl = (
            self.temp_root / "train.jsonl"
        )
        self.test_jsonl = (
            self.temp_root / "test.jsonl"
        )

        train_rows = [
            {
                "index": 0,
                "split": "train",
                "image_id": "synpic_large",
                "image_name": "synpic_large.jpg",
                "image_path": str(
                    self.large_image_path
                ),
                "question": (
                    "is this image normal?"
                ),
                "answer": "yes",
                "references": ["yes"],
                "answer_type": "CLOSED",
            },
            {
                "index": 1,
                "split": "train",
                "image_id": "synpic_small",
                "image_name": "synpic_small.jpg",
                "image_path": str(
                    self.small_image_path
                ),
                "question": (
                    "what abnormality is seen?"
                ),
                "answer": (
                    "carcinoma, small cell."
                ),
                "references": [
                    "carcinoma, small cell."
                ],
                "answer_type": "OPEN",
            },
        ]

        test_rows = [
            {
                "index": 0,
                "split": "test",
                "image_id": "synpic_large",
                "image_name": "synpic_large.jpg",
                "image_path": str(
                    self.large_image_path
                ),
                "question": (
                    "what is the primary abnormality?"
                ),
                "answer": (
                    "avascular necrosis"
                ),
                "references": [
                    "avascular necrosis",
                    (
                        "REFERENCE_ONLY_NEVER_IN_PROMPT "
                        "bilateral femoral head disease"
                    ),
                ],
                "answer_type": "OPEN",
            },
            {
                "index": 1,
                "split": "test",
                "image_id": "synpic_small",
                "image_name": "synpic_small.jpg",
                "image_path": str(
                    self.small_image_path
                ),
                "question": (
                    "what is abnormal in the image?"
                ),
                "answer": "osteopoikilosis",
                "references": [
                    "osteopoikilosis",
                    "osteosclerotic lesions",
                ],
                "answer_type": "OPEN",
            },
        ]

        self._write_jsonl(
            self.train_jsonl,
            train_rows,
        )
        self._write_jsonl(
            self.test_jsonl,
            test_rows,
        )

        self.train_dataset = (
            VQAMED2021Dataset(
                jsonl_path=str(
                    self.train_jsonl
                ),
                expected_split="train",
                expected_count=2,
                verify_images=True,
                strict=True,
            )
        )

        self.test_dataset = (
            VQAMED2021Dataset(
                jsonl_path=str(
                    self.test_jsonl
                ),
                expected_split="test",
                expected_count=2,
                verify_images=True,
                strict=True,
            )
        )

        self.train_batch = [
            self.train_dataset[0],
            self.train_dataset[1],
        ]
        self.test_batch = [
            self.test_dataset[0],
            self.test_dataset[1],
        ]

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    @staticmethod
    def _write_jsonl(
        path: Path,
        rows: Sequence[dict[str, Any]],
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for row in rows:
                file.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _assert_qwen_images_resized(
        self,
        processor: FakeProcessor,
    ) -> None:
        self.assertGreaterEqual(
            len(processor.call_records),
            1,
        )

        for record in processor.call_records:
            for width, height in record[
                "image_sizes"
            ]:
                self.assertLessEqual(
                    max(width, height),
                    64,
                )

        # 大图必须缩放为 64×32，小图保持 40×50。
        last_sizes = processor.call_records[
            -1
        ]["image_sizes"]

        self.assertEqual(
            last_sizes[0],
            (64, 32),
        )
        self.assertEqual(
            last_sizes[1],
            (40, 50),
        )

    def test_train_a0_masks_prompt_and_excludes_biomed(
        self,
    ) -> None:
        processor = FakeProcessor(
            padding_side="right"
        )
        cfg = make_cfg(
            enable_visual_adapter=False
        )

        original_batch = copy.deepcopy(
            self.train_batch
        )

        collator = VQAMED2021TrainCollator(
            processor=processor,
            cfg=cfg,
        )

        batch_inputs = collator(
            self.train_batch
        )

        self.assertEqual(
            self.train_batch,
            original_batch,
            "Train Collator 不应原地修改 Dataset 样本。",
        )

        required_keys = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "labels",
        }
        self.assertTrue(
            required_keys.issubset(
                batch_inputs.keys()
            )
        )

        self.assertNotIn(
            "biomed_image_tensors",
            batch_inputs,
        )
        self.assertNotIn(
            "biomed_text_tokens",
            batch_inputs,
        )

        self.assertEqual(
            len(processor.call_records),
            2,
            "Train Collator 应分别处理 prompt 与 full sequence。",
        )

        prompt_record = (
            processor.call_records[0]
        )
        full_record = (
            processor.call_records[1]
        )

        labels = batch_inputs["labels"]
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs[
            "attention_mask"
        ]

        self.assertEqual(
            tuple(labels.shape),
            tuple(input_ids.shape),
        )

        for sample_index in range(
            len(self.train_batch)
        ):
            prompt_length = int(
                prompt_record[
                    "attention_mask"
                ][sample_index].sum().item()
            )
            full_length = int(
                attention_mask[
                    sample_index
                ].sum().item()
            )

            # Prompt 全部被 mask。
            self.assertTrue(
                torch.all(
                    labels[
                        sample_index,
                        :prompt_length,
                    ].eq(-100)
                )
            )

            # Assistant 答案至少有一个监督 token。
            self.assertGreater(
                int(
                    labels[
                        sample_index,
                        prompt_length:full_length,
                    ]
                    .ne(-100)
                    .sum()
                    .item()
                ),
                0,
            )

            # Padding 也必须被 mask。
            if full_length < labels.shape[1]:
                self.assertTrue(
                    torch.all(
                        labels[
                            sample_index,
                            full_length:,
                        ].eq(-100)
                    )
                )

        prompt_text = "\n".join(
            prompt_record["texts"]
        )
        full_text = "\n".join(
            full_record["texts"]
        )

        self.assertIn(
            "QWEN_ONLY_INSTRUCTION",
            prompt_text,
        )
        self.assertNotIn(
            "carcinoma, small cell",
            prompt_text,
            "训练 Prompt 本身不能包含答案。",
        )
        self.assertIn(
            "carcinoma, small cell",
            full_text,
            "完整训练序列必须包含清洗后的答案。",
        )

        self._assert_qwen_images_resized(
            processor
        )

    def test_train_a6_adds_biomed_and_uses_raw_questions(
        self,
    ) -> None:
        processor = FakeProcessor(
            padding_side="right"
        )
        transform = FakeBioMedTransform()
        tokenizer = FakeBioMedTokenizer()
        cfg = make_cfg(
            enable_visual_adapter=True
        )

        collator = VQAMED2021TrainCollator(
            processor=processor,
            cfg=cfg,
            biomed_transform=transform,
            biomed_tokenizer=tokenizer,
        )

        batch_inputs = collator(
            self.train_batch
        )

        self.assertIn(
            "biomed_image_tensors",
            batch_inputs,
        )
        self.assertIn(
            "biomed_text_tokens",
            batch_inputs,
        )

        self.assertEqual(
            tuple(
                batch_inputs[
                    "biomed_image_tensors"
                ].shape
            ),
            (2, 3, 8, 8),
        )
        self.assertEqual(
            tuple(
                batch_inputs[
                    "biomed_text_tokens"
                ].shape
            ),
            (2, 8),
        )

        self.assertEqual(
            tokenizer.calls,
            [[
                "is this image normal?",
                "what abnormality is seen?",
            ]],
            "BioMedCLIP Router 必须读取原始 question。",
        )

        for question in tokenizer.calls[0]:
            self.assertNotIn(
                "QWEN_ONLY_INSTRUCTION",
                question,
            )

        # BioMedCLIP 应收到原始图像，而不是 Qwen 缩放后的图像。
        self.assertEqual(
            transform.image_sizes,
            [
                (160, 80),
                (40, 50),
            ],
        )

        self.assertEqual(
            batch_inputs[
                "biomed_text_tokens"
            ].device.type,
            "cpu",
        )

    def test_eval_a0_preserves_references_and_prevents_leakage(
        self,
    ) -> None:
        processor = FakeProcessor(
            padding_side="left"
        )
        cfg = make_cfg(
            enable_visual_adapter=False
        )

        original_batch = copy.deepcopy(
            self.test_batch
        )

        collator = VQAMED2021EvalCollator(
            processor=processor,
            cfg=cfg,
        )

        batch_inputs, metadata_list = (
            collator(self.test_batch)
        )

        self.assertEqual(
            self.test_batch,
            original_batch,
            "Eval Collator 不应原地修改 Dataset 样本。",
        )

        self.assertNotIn(
            "labels",
            batch_inputs,
        )
        self.assertNotIn(
            "biomed_image_tensors",
            batch_inputs,
        )
        self.assertNotIn(
            "biomed_text_tokens",
            batch_inputs,
        )

        self.assertEqual(
            len(metadata_list),
            2,
        )

        first_meta = metadata_list[0]

        self.assertEqual(
            first_meta["index"],
            0,
        )
        self.assertEqual(
            first_meta["split"],
            "test",
        )
        self.assertEqual(
            first_meta["image_id"],
            "synpic_large",
        )
        self.assertEqual(
            first_meta["image_name"],
            "synpic_large.jpg",
        )
        self.assertEqual(
            first_meta["question"],
            "what is the primary abnormality?",
        )
        self.assertEqual(
            first_meta["gt_answer"],
            "avascular necrosis",
        )
        self.assertEqual(
            first_meta["references"],
            [
                "avascular necrosis",
                (
                    "REFERENCE_ONLY_NEVER_IN_PROMPT "
                    "bilateral femoral head disease"
                ),
            ],
        )
        self.assertEqual(
            first_meta["question_type"],
            "Abnormality",
        )
        self.assertEqual(
            first_meta["answer_type"],
            "OPEN",
        )

        prompt_text = "\n".join(
            processor.call_records[-1][
                "texts"
            ]
        )

        self.assertIn(
            "what is the primary abnormality?",
            prompt_text,
        )
        self.assertIn(
            "QWEN_ONLY_INSTRUCTION",
            prompt_text,
        )

        # Test 的 answer/references 只能进入 metadata，
        # 不能进入模型 Prompt。
        forbidden_texts = [
            "avascular necrosis",
            "REFERENCE_ONLY_NEVER_IN_PROMPT",
            "osteopoikilosis",
            "osteosclerotic lesions",
        ]
        for forbidden in forbidden_texts:
            self.assertNotIn(
                forbidden,
                prompt_text,
                f"发现答案泄漏到 Eval Prompt：{forbidden!r}",
            )

        self._assert_qwen_images_resized(
            processor
        )

    def test_eval_a6_adds_biomed_and_uses_raw_questions(
        self,
    ) -> None:
        processor = FakeProcessor(
            padding_side="left"
        )
        transform = FakeBioMedTransform()
        tokenizer = FakeBioMedTokenizer()
        cfg = make_cfg(
            enable_visual_adapter=True
        )

        collator = VQAMED2021EvalCollator(
            processor=processor,
            cfg=cfg,
            biomed_transform=transform,
            biomed_tokenizer=tokenizer,
        )

        batch_inputs, metadata_list = (
            collator(self.test_batch)
        )

        self.assertEqual(
            len(metadata_list),
            2,
        )
        self.assertEqual(
            tuple(
                batch_inputs[
                    "biomed_image_tensors"
                ].shape
            ),
            (2, 3, 8, 8),
        )
        self.assertEqual(
            tuple(
                batch_inputs[
                    "biomed_text_tokens"
                ].shape
            ),
            (2, 8),
        )

        self.assertEqual(
            tokenizer.calls,
            [[
                "what is the primary abnormality?",
                "what is abnormal in the image?",
            ]],
        )

        self.assertEqual(
            transform.image_sizes,
            [
                (160, 80),
                (40, 50),
            ],
        )

    def test_padding_side_contracts(
        self,
    ) -> None:
        cfg_a0 = make_cfg(
            enable_visual_adapter=False
        )

        with self.assertRaises(ValueError):
            VQAMED2021TrainCollator(
                processor=FakeProcessor(
                    padding_side="left"
                ),
                cfg=cfg_a0,
            )

        with self.assertRaises(ValueError):
            VQAMED2021EvalCollator(
                processor=FakeProcessor(
                    padding_side="right"
                ),
                cfg=cfg_a0,
            )

    def test_visual_adapter_requires_biomed_dependencies(
        self,
    ) -> None:
        cfg_a6 = make_cfg(
            enable_visual_adapter=True
        )

        with self.assertRaises(ValueError):
            VQAMED2021TrainCollator(
                processor=FakeProcessor(
                    padding_side="right"
                ),
                cfg=cfg_a6,
            )

        with self.assertRaises(ValueError):
            VQAMED2021EvalCollator(
                processor=FakeProcessor(
                    padding_side="left"
                ),
                cfg=cfg_a6,
            )

    def test_empty_batch_is_rejected(
        self,
    ) -> None:
        cfg_a0 = make_cfg(
            enable_visual_adapter=False
        )

        train_collator = (
            VQAMED2021TrainCollator(
                processor=FakeProcessor(
                    padding_side="right"
                ),
                cfg=cfg_a0,
            )
        )
        eval_collator = (
            VQAMED2021EvalCollator(
                processor=FakeProcessor(
                    padding_side="left"
                ),
                cfg=cfg_a0,
            )
        )

        with self.assertRaises(ValueError):
            train_collator([])

        with self.assertRaises(ValueError):
            eval_collator([])


if __name__ == "__main__":
    print("=" * 80)
    print("VQA-Med 2021 Collator 测试")
    print(f"Project root: {PROJECT_ROOT}")
    print(
        "本测试使用 FakeProcessor，不会加载 Qwen3-VL 或 BioMedCLIP 大模型。"
    )
    print("=" * 80)

    unittest.main(
        verbosity=2,
    )