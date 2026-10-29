#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate JSONL manifests for ImageCLEF VQA-Med 2021 Task 1.

Outputs:
  train.jsonl             4500 rows, with answer/references
  validation.jsonl         500 rows, with answer/references
  test.jsonl               500 rows, with answer/references
  dataset_summary.json        dataset statistics

All splits use one unified schema. During inference, only image_path and
question are used as model input; answer/references are used only by the
evaluator.

The source annotation order is preserved.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


# ============================================================
# 1. Paths
# ============================================================

DATA_ROOT = Path("/home/yuqing/Datas/VQA-Med-2021")

TRAIN_IMAGES = (
    DATA_ROOT / "SYSU-HCP/extracted/data/train/images"
)
TRAIN_QA = (
    DATA_ROOT / "SYSU-HCP/extracted/data/train/label.txt"
)

VAL_ROOT = (
    DATA_ROOT
    / "extracted/validation_2021"
    / "VQA-Med-2021-Tasks-1-2-NewValidationSets"
)
VAL_IMAGES = (
    VAL_ROOT / "ImageCLEF-2021-VQA-Med-New-Validation-Images"
)
VAL_QA = (
    VAL_ROOT
    / "VQA-Med-2021-VQAnswering-Task1-New-ValidationSet.txt"
)

TEST_ROOT = (
    DATA_ROOT
    / "extracted/test_2021"
    / "Task1-VQA-2021-TestSet-w-GroundTruth"
)

# Test images are already extracted here.
TEST_IMAGES = (
    DATA_ROOT / "SYSU-HCP/extracted/data/test2021/images"
)

# Questions and references use the official files.
TEST_QUESTIONS = (
    TEST_ROOT / "Task1-VQA-2021-TestSet-Questions.txt"
)
TEST_REFERENCES = (
    TEST_ROOT / "Task1-VQA-2021-TestSet-ReferenceAnswers.txt"
)

OUTPUT_DIR = DATA_ROOT / "jsonl"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ============================================================
# 2. General utilities
# ============================================================

def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist:\n  {path}"
        )


def iter_nonempty_lines(path: Path) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line:
                yield line_number, line


def build_image_index(image_root: Path) -> Dict[str, Path]:
    """
    Build:
        image_id -> absolute image path

    Duplicate stems are rejected, because annotations identify images by stem.
    """
    require_path(image_root, "Image directory")

    image_index: Dict[str, Path] = {}
    duplicate_ids: Dict[str, List[str]] = {}

    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    for image_path in image_paths:
        image_id = image_path.stem.strip()

        if image_id in image_index:
            duplicate_ids.setdefault(
                image_id, [str(image_index[image_id])]
            ).append(str(image_path))
        else:
            image_index[image_id] = image_path.resolve()

    if duplicate_ids:
        preview = "\n".join(
            f"  {image_id}: {paths}"
            for image_id, paths in list(duplicate_ids.items())[:10]
        )
        raise ValueError(
            "Duplicate image IDs were found. Image stems must be unique.\n"
            + preview
        )

    return image_index


def answer_type(answer: str) -> str:
    return "CLOSED" if answer.strip().lower() in {"yes", "no"} else "OPEN"


def write_jsonl(rows: Iterable[Mapping[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1

    return count


def assert_unique_ids(rows: List[Dict[str, Any]], split: str) -> None:
    ids = [row["image_id"] for row in rows]
    duplicates = [
        image_id
        for image_id, count in Counter(ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(
            f"{split}: duplicate image IDs found: {duplicates[:20]}"
        )


# ============================================================
# 3. Parsers
# ============================================================

def read_qa_rows(
    annotation_path: Path,
    image_index: Mapping[str, Path],
    split: str,
) -> List[Dict[str, Any]]:
    """
    Parse:
        image_id|question|answer
    """
    require_path(annotation_path, f"{split} annotation file")

    rows: List[Dict[str, Any]] = []
    missing_images: List[str] = []

    for index, (line_number, line) in enumerate(
        iter_nonempty_lines(annotation_path)
    ):
        parts = line.split("|", 2)
        if len(parts) != 3:
            raise ValueError(
                f"{annotation_path}:{line_number}: expected "
                "'image_id|question|answer', got:\n{line}"
            )

        image_id, question, answer = (
            part.strip() for part in parts
        )

        if not image_id or not question or not answer:
            raise ValueError(
                f"{annotation_path}:{line_number}: empty field found:\n{line}"
            )

        image_path = image_index.get(image_id)
        if image_path is None:
            missing_images.append(image_id)
            continue

        rows.append(
            {
                "index": index,
                "split": split,
                "image_id": image_id,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "question": question,
                "answer": answer,
                "references": [answer],
                "answer_type": answer_type(answer),
            }
        )

    if missing_images:
        raise FileNotFoundError(
            f"{split}: {len(missing_images)} referenced images are missing. "
            f"Examples: {missing_images[:20]}"
        )

    assert_unique_ids(rows, split)
    return rows


def read_test_questions(
    questions_path: Path,
    image_index: Mapping[str, Path],
) -> List[Dict[str, Any]]:
    """
    Parse:
        image_id|question

    Ground-truth answers are attached later after the official reference
    file is matched by image_id.
    """
    require_path(questions_path, "Official test questions file")

    rows: List[Dict[str, Any]] = []
    missing_images: List[str] = []

    for index, (line_number, line) in enumerate(
        iter_nonempty_lines(questions_path)
    ):
        parts = line.split("|", 1)
        if len(parts) != 2:
            raise ValueError(
                f"{questions_path}:{line_number}: expected "
                "'image_id|question', got:\n{line}"
            )

        image_id, question = (part.strip() for part in parts)

        if not image_id or not question:
            raise ValueError(
                f"{questions_path}:{line_number}: empty field found:\n{line}"
            )

        image_path = image_index.get(image_id)
        if image_path is None:
            missing_images.append(image_id)
            continue

        rows.append(
            {
                "index": index,
                "split": "test",
                "image_id": image_id,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "question": question,
            }
        )

    if missing_images:
        raise FileNotFoundError(
            f"test: {len(missing_images)} referenced images are missing. "
            f"Examples: {missing_images[:20]}"
        )

    assert_unique_ids(rows, "test")
    return rows


def read_test_references(
    references_path: Path,
) -> Dict[str, List[str]]:
    """
    Parse:
        image_id|reference_1|reference_2|...

    Every non-empty field after image_id is retained as one acceptable
    reference answer.
    """
    require_path(references_path, "Official test references file")

    reference_map: Dict[str, List[str]] = {}

    for line_number, line in iter_nonempty_lines(references_path):
        parts = [part.strip() for part in line.split("|")]

        if len(parts) < 2:
            raise ValueError(
                f"{references_path}:{line_number}: expected at least "
                "'image_id|reference', got:\n{line}"
            )

        image_id = parts[0]
        references = [part for part in parts[1:] if part]

        if not image_id or not references:
            raise ValueError(
                f"{references_path}:{line_number}: empty image ID or "
                f"no valid references:\n{line}"
            )

        if image_id in reference_map:
            raise ValueError(
                f"{references_path}:{line_number}: duplicate image ID "
                f"{image_id!r}"
            )

        reference_map[image_id] = references

    return reference_map


# ============================================================
# 4. Main
# ============================================================

def main() -> None:
    print("Building image indexes...")
    train_image_index = build_image_index(TRAIN_IMAGES)
    val_image_index = build_image_index(VAL_IMAGES)
    test_image_index = build_image_index(TEST_IMAGES)

    print("Reading annotations...")
    train_rows = read_qa_rows(
        TRAIN_QA, train_image_index, split="train"
    )
    val_rows = read_qa_rows(
        VAL_QA, val_image_index, split="validation"
    )
    test_rows = read_test_questions(
        TEST_QUESTIONS, test_image_index
    )
    test_reference_map = read_test_references(TEST_REFERENCES)

    # Keep test references in exactly the same order as test questions.
    test_ids = [row["image_id"] for row in test_rows]
    question_id_set = set(test_ids)
    reference_id_set = set(test_reference_map)

    missing_references = question_id_set - reference_id_set
    extra_references = reference_id_set - question_id_set

    if missing_references or extra_references:
        raise ValueError(
            "Test question/reference IDs do not match.\n"
            f"Missing references: {sorted(missing_references)[:20]}\n"
            f"Extra references: {sorted(extra_references)[:20]}"
        )

    # Merge official test references into test.jsonl.
    # "answer" is the first/canonical official reference.
    # "references" contains every acceptable official reference.
    for row in test_rows:
        references = test_reference_map[row["image_id"]]
        row["answer"] = references[0]
        row["references"] = references
        row["answer_type"] = answer_type(references[0])

    # Freeze expected official split sizes.
    expected_counts = {
        "train": 4500,
        "validation": 500,
        "test": 500,
    }
    actual_counts = {
        "train": len(train_rows),
        "validation": len(val_rows),
        "test": len(test_rows),
    }

    if actual_counts != expected_counts:
        raise AssertionError(
            "Unexpected split sizes.\n"
            f"Expected: {expected_counts}\n"
            f"Actual:   {actual_counts}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train": OUTPUT_DIR / "train.jsonl",
        "validation": OUTPUT_DIR / "validation.jsonl",
        "test": OUTPUT_DIR / "test.jsonl",
    }

    written_counts = {
        "train": write_jsonl(train_rows, output_paths["train"]),
        "validation": write_jsonl(
            val_rows, output_paths["validation"]
        ),
        "test": write_jsonl(test_rows, output_paths["test"]),
    }

    summary = {
        "data_root": str(DATA_ROOT.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "source_image_counts": {
            "train_directory": len(train_image_index),
            "validation_directory": len(val_image_index),
            "test_directory": len(test_image_index),
        },
        "jsonl_counts": written_counts,
        "closed_open_counts": {
            "train": dict(
                Counter(row["answer_type"] for row in train_rows)
            ),
            "validation": dict(
                Counter(row["answer_type"] for row in val_rows)
            ),
            "test": dict(
                Counter(row["answer_type"] for row in test_rows)
            ),
        },
        "files": {
            name: str(path.resolve())
            for name, path in output_paths.items()
        },
    }

    summary_path = OUTPUT_DIR / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n===== JSONL generation complete =====")
    print(f"Train           : {written_counts['train']:4d} -> {output_paths['train']}")
    print(
        f"Validation      : {written_counts['validation']:4d} -> "
        f"{output_paths['validation']}"
    )
    print(f"Test            : {written_counts['test']:4d} -> {output_paths['test']}")
    print(f"Summary          : {summary_path}")
    print("\nPASS: VQA-Med 2021 JSONL files are ready.")


if __name__ == "__main__":
    main()