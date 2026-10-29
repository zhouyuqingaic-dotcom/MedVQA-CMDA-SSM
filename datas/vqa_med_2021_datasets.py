from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


class VQAMED2021Dataset(Dataset):
    """
    VQA-Med 2021 Task 1 JSONL 数据集读取类。

    设计目标
    --------
    1. 只负责读取、校验和标准化样本，不负责：
       - Prompt 构造
       - 图像预处理
       - Tokenize
       - 答案清洗
       - 训练/评测逻辑

    2. 统一支持：
       - train.jsonl
       - validation.jsonl
       - test.jsonl

    3. 保留 VQA-Med 2021 Test 的多参考答案：
       - answer: 主参考答案，训练时可作为监督目标
       - references: 全部官方参考答案，评测时使用

    4. JSONL 中虽然包含 Test 答案，但 Dataset 不会自动把答案送进模型。
       是否使用 answer/references，由 Train/Eval Collator 决定。

    当前 JSONL 单条样本格式
    ----------------------
    {
        "index": 0,
        "split": "test",
        "image_id": "synpic42072",
        "image_name": "synpic42072.jpg",
        "image_path": "/absolute/path/synpic42072.jpg",
        "question": "what abnormality is seen in the image?",
        "answer": "avascular necrosis",
        "references": [
            "avascular necrosis",
            "extensive degenerative changes ..."
        ],
        "answer_type": "OPEN"
    }

    返回给 Collator 的字段
    ---------------------
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
        "raw_row": dict
    }
    """

    VALID_SPLITS = {"train", "validation", "test"}
    VALID_ANSWER_TYPES = {"OPEN", "CLOSED"}

    def __init__(
        self,
        jsonl_path: str,
        expected_split: Optional[str] = None,
        expected_count: Optional[int] = None,
        verify_images: bool = False,
        strict: bool = True,
        image_root: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        jsonl_path:
            train.jsonl、validation.jsonl 或 test.jsonl 路径。

        expected_split:
            可选。指定后会检查每条数据的 split 字段。
            推荐分别传入：
                train
                validation
                test

        expected_count:
            可选。指定后会检查样本总数。
            当前官方划分推荐：
                train: 4500
                validation: 500
                test: 500

        verify_images:
            是否在 Dataset 初始化时检查全部图片是否存在。
            多卡训练时每个进程都会初始化 Dataset，因此正式训练可设为 False；
            首次检查数据时可设为 True。

        strict:
            是否启用严格字段检查。
            推荐保持 True，遇到坏数据直接报错，不静默跳过。

        image_root:
            可选。仅在 JSONL 的 image_path 是相对路径时使用。
            当前生成的 JSONL 使用绝对路径，所以通常不需要传入。
        """
        self.jsonl_path = Path(jsonl_path).expanduser()
        self.expected_split = self._normalize_expected_split(expected_split)
        self.expected_count = expected_count
        self.verify_images = bool(verify_images)
        self.strict = bool(strict)
        self.image_root = (
            Path(image_root).expanduser()
            if image_root is not None
            else None
        )

        if not self.jsonl_path.is_file():
            raise FileNotFoundError(
                f"VQA-Med 2021 JSONL 文件不存在: {self.jsonl_path}"
            )

        if self.image_root is not None and not self.image_root.is_dir():
            raise FileNotFoundError(
                f"VQA-Med 2021 图像根目录不存在: {self.image_root}"
            )

        if self.expected_count is not None:
            if not isinstance(self.expected_count, int):
                raise TypeError(
                    "expected_count 必须是 int 或 None，"
                    f"当前类型为 {type(self.expected_count)}"
                )
            if self.expected_count < 0:
                raise ValueError(
                    f"expected_count 不能小于 0: {self.expected_count}"
                )

        self.samples: List[Dict[str, Any]] = self._read_jsonl(
            self.jsonl_path
        )

        if self.expected_count is not None:
            actual_count = len(self.samples)
            if actual_count != self.expected_count:
                raise ValueError(
                    "VQA-Med 2021 样本数量不符合预期："
                    f"path={self.jsonl_path}, "
                    f"expected={self.expected_count}, "
                    f"actual={actual_count}"
                )

        if self.verify_images:
            self._verify_all_images()

    @classmethod
    def _normalize_expected_split(
        cls,
        expected_split: Optional[str],
    ) -> Optional[str]:
        if expected_split is None:
            return None

        split = str(expected_split).strip().lower()

        split_aliases = {
            "val": "validation",
            "valid": "validation",
            "dev": "validation",
        }
        split = split_aliases.get(split, split)

        if split not in cls.VALID_SPLITS:
            raise ValueError(
                "expected_split 必须是 train、validation、test 或 None，"
                f"当前值为 {expected_split!r}"
            )

        return split

    def _read_jsonl(
        self,
        path: Path,
    ) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        with path.open("r", encoding="utf-8-sig") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "VQA-Med 2021 JSONL 解析失败："
                        f"path={path}, line={line_number}, "
                        f"error={exc}"
                    ) from exc

                if not isinstance(row, dict):
                    raise TypeError(
                        "VQA-Med 2021 JSONL 每行必须是 JSON object："
                        f"path={path}, line={line_number}, "
                        f"当前类型={type(row)}"
                    )

                sample = self._normalize_row(
                    row=row,
                    line_number=line_number,
                )
                samples.append(sample)

        if not samples:
            raise ValueError(
                f"VQA-Med 2021 JSONL 中没有有效样本: {path}"
            )

        return samples

    def _normalize_row(
        self,
        row: Dict[str, Any],
        line_number: int,
    ) -> Dict[str, Any]:
        context = (
            f"path={self.jsonl_path}, line={line_number}"
        )

        # --------------------------------------------------
        # 1. index
        # --------------------------------------------------
        raw_index = row.get("index", line_number - 1)

        try:
            source_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"index 必须能转换为 int：{context}, "
                f"value={raw_index!r}"
            ) from exc

        if self.strict and source_index < 0:
            raise ValueError(
                f"index 不能小于 0：{context}, index={source_index}"
            )

        # --------------------------------------------------
        # 2. split
        # --------------------------------------------------
        raw_split = row.get(
            "split",
            self.expected_split if self.expected_split is not None else "",
        )
        split = str(raw_split).strip().lower()

        split_aliases = {
            "val": "validation",
            "valid": "validation",
            "dev": "validation",
        }
        split = split_aliases.get(split, split)

        if not split:
            raise ValueError(
                f"样本缺少 split 字段：{context}"
            )

        if self.strict and split not in self.VALID_SPLITS:
            raise ValueError(
                f"非法 split：{context}, split={split!r}"
            )

        if (
            self.expected_split is not None
            and split != self.expected_split
        ):
            raise ValueError(
                "JSONL 中的 split 与 expected_split 不一致："
                f"{context}, row_split={split!r}, "
                f"expected_split={self.expected_split!r}"
            )

        # --------------------------------------------------
        # 3. image_path / image_name / image_id
        # --------------------------------------------------
        raw_image_path = str(
            row.get("image_path", "")
        ).strip()

        if not raw_image_path:
            raise ValueError(
                f"样本缺少 image_path：{context}"
            )

        image_path = Path(raw_image_path).expanduser()

        if not image_path.is_absolute():
            if self.image_root is not None:
                image_path = self.image_root / image_path
            else:
                # 相对路径默认相对于 JSONL 所在目录。
                image_path = self.jsonl_path.parent / image_path

        # strict=False 时也不要求文件此刻存在，因此使用 strict=False。
        image_path = image_path.resolve(strict=False)

        image_name = str(
            row.get("image_name", image_path.name)
        ).strip()

        if not image_name:
            image_name = image_path.name

        image_id = str(
            row.get("image_id", image_path.stem)
        ).strip()

        if not image_id:
            image_id = image_path.stem

        if self.strict:
            if not image_name:
                raise ValueError(
                    f"无法确定 image_name：{context}"
                )
            if not image_id:
                raise ValueError(
                    f"无法确定 image_id：{context}"
                )

            # 当前 JSONL 应保持 image_name 与 image_path 文件名一致。
            if image_name != image_path.name:
                raise ValueError(
                    "image_name 与 image_path 的文件名不一致："
                    f"{context}, image_name={image_name!r}, "
                    f"image_path_name={image_path.name!r}"
                )

            # 当前 JSONL 应保持 image_id 与文件 stem 一致。
            if image_id != image_path.stem:
                raise ValueError(
                    "image_id 与 image_path 的 stem 不一致："
                    f"{context}, image_id={image_id!r}, "
                    f"image_stem={image_path.stem!r}"
                )

        # --------------------------------------------------
        # 4. question
        # --------------------------------------------------
        question = str(
            row.get("question", "")
        ).strip()

        if not question:
            raise ValueError(
                f"样本问题为空：{context}, image_id={image_id}"
            )

        # --------------------------------------------------
        # 5. answer
        # --------------------------------------------------
        answer = str(
            row.get("answer", "")
        ).strip()

        if not answer:
            raise ValueError(
                f"样本答案为空：{context}, image_id={image_id}"
            )

        # --------------------------------------------------
        # 6. references
        # --------------------------------------------------
        raw_references = row.get("references")

        if raw_references is None:
            references = [answer]
        elif isinstance(raw_references, str):
            references = [raw_references.strip()]
        elif isinstance(raw_references, (list, tuple)):
            references = [
                str(reference).strip()
                for reference in raw_references
                if str(reference).strip()
            ]
        else:
            raise TypeError(
                "references 必须是 list、tuple、str 或 None："
                f"{context}, 当前类型={type(raw_references)}"
            )

        # 按原顺序去重，但不改变文本大小写和医学表达。
        references = list(dict.fromkeys(references))

        if not references:
            raise ValueError(
                f"references 为空：{context}, image_id={image_id}"
            )

        if self.strict and answer not in references:
            raise ValueError(
                "主答案 answer 不在 references 中："
                f"{context}, image_id={image_id}, "
                f"answer={answer!r}, references={references!r}"
            )

        # --------------------------------------------------
        # 7. answer_type
        # --------------------------------------------------
        inferred_answer_type = (
            "CLOSED"
            if answer.lower() in {"yes", "no"}
            else "OPEN"
        )

        raw_answer_type = row.get(
            "answer_type",
            inferred_answer_type,
        )
        answer_type = str(raw_answer_type).strip().upper()

        if answer_type not in self.VALID_ANSWER_TYPES:
            raise ValueError(
                "answer_type 必须是 OPEN 或 CLOSED："
                f"{context}, value={raw_answer_type!r}"
            )

        if self.strict and answer_type != inferred_answer_type:
            raise ValueError(
                "answer_type 与主答案内容不一致："
                f"{context}, answer={answer!r}, "
                f"answer_type={answer_type!r}, "
                f"inferred={inferred_answer_type!r}"
            )

        # --------------------------------------------------
        # 8. question_type
        # --------------------------------------------------
        # VQA-Med 2021 Task 1 是 Abnormality 问答。
        question_type = str(
            row.get("question_type", "Abnormality")
        ).strip()

        if not question_type:
            question_type = "Abnormality"

        # raw_row 保留 JSONL 原始字段，方便后续调试与追踪。
        raw_row = dict(row)

        return {
            "index": source_index,
            "split": split,
            "image_id": image_id,
            "image_name": image_name,
            "image_path": str(image_path),
            "question": question,
            "answer": answer,
            "references": references,
            "question_type": question_type,
            "answer_type": answer_type,
            "raw_row": raw_row,
        }

    def _verify_all_images(self) -> None:
        missing_images: List[Dict[str, Any]] = []

        for dataset_index, sample in enumerate(self.samples):
            image_path = Path(sample["image_path"])

            if not image_path.is_file():
                missing_images.append(
                    {
                        "dataset_index": dataset_index,
                        "source_index": sample["index"],
                        "image_id": sample["image_id"],
                        "image_path": str(image_path),
                    }
                )

        if missing_images:
            preview = "\n".join(
                (
                    f"  dataset_index={item['dataset_index']}, "
                    f"source_index={item['source_index']}, "
                    f"image_id={item['image_id']}, "
                    f"image_path={item['image_path']}"
                )
                for item in missing_images[:10]
            )

            raise FileNotFoundError(
                "VQA-Med 2021 存在缺失图片："
                f"missing={len(missing_images)}, "
                f"jsonl={self.jsonl_path}\n"
                f"前 10 个缺失样本：\n{preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        sample = self.samples[index]

        # 返回新的浅拷贝，避免 Collator 意外修改 Dataset 内部缓存。
        return {
            "index": sample["index"],
            "split": sample["split"],
            "image_id": sample["image_id"],
            "image_name": sample["image_name"],
            "image_path": sample["image_path"],
            "question": sample["question"],
            "answer": sample["answer"],
            "references": list(sample["references"]),
            "question_type": sample["question_type"],
            "answer_type": sample["answer_type"],
            "raw_row": dict(sample["raw_row"]),
        }

    def __repr__(self) -> str:
        split = (
            self.expected_split
            if self.expected_split is not None
            else "mixed/unspecified"
        )

        return (
            f"{self.__class__.__name__}("
            f"jsonl_path={str(self.jsonl_path)!r}, "
            f"split={split!r}, "
            f"num_samples={len(self)}, "
            f"verify_images={self.verify_images}, "
            f"strict={self.strict}"
            f")"
        )