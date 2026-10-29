"""
生成一个与 MIMICCXRDataset 一级 valid-indices 缓存格式完全相同的固定子集缓存。

目标：
1. 读取 stage1_train_config_mimic_cxr.py 中配置的完整有效缓存；
2. 从完整有效样本中固定抽取 N 条；
3. 输出仍然是纯整数列表 JSON，与原 valid_indices.json 格式一致；
4. Stage 1 Trainer 无需修改，只需切换 TrainConfig.mimic_cxr_cache_prefix。

重要：
- 输出列表中的索引是相对于原始 filtered metadata 的索引，
  不是相对于 320016 条有效样本的局部索引。
- 因此新缓存可以被 MIMICCXRDataset 原样加载。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _normalize_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _size_label(size: int) -> str:
    if size % 1000 == 0:
        return f"{size // 1000}k"
    return str(size)


def _load_index_list(path: Path) -> List[int]:
    if not path.is_file():
        raise FileNotFoundError(f"缓存不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"缓存顶层必须是 list，但当前为 {type(data).__name__}：{path}"
        )

    if not all(
        isinstance(index, int) and not isinstance(index, bool)
        for index in data
    ):
        raise ValueError(f"缓存必须是纯整数索引列表：{path}")

    if len(data) != len(set(data)):
        raise ValueError(f"缓存中存在重复索引：{path}")

    return data


def _save_index_list(path: Path, indices: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(list(indices), file, ensure_ascii=False)

    os.replace(temp_path, path)


def _sha256_indices(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{index}\n".encode("utf-8"))
    return digest.hexdigest()


def _select_image_positions(
    dataset,
    subset_size: int,
    subset_seed: int,
) -> List[int]:
    positions = list(range(len(dataset)))
    random.Random(subset_seed).shuffle(positions)
    return positions[:subset_size]


def _select_study_one_image_positions(
    dataset,
    subset_size: int,
    subset_seed: int,
) -> List[int]:
    samples = dataset.samples

    subject_column = getattr(
        dataset,
        "subject_id_column",
        "subject_id",
    )
    study_column = getattr(
        dataset,
        "study_id_column",
        "study_id",
    )
    dicom_column = getattr(
        dataset,
        "dicom_id_column",
        "dicom_id",
    )

    study_to_positions: Dict[Tuple[str, str], List[int]] = {}

    for position, row in enumerate(samples):
        subject_id = _normalize_id(
            row.get(subject_column, "")
        )
        study_id = _normalize_id(
            row.get(study_column, "")
        )
        dicom_id = str(
            row.get(dicom_column, "")
        ).strip()

        if not subject_id or not study_id or not dicom_id:
            raise ValueError(
                "发现缺少 subject_id / study_id / dicom_id 的样本："
                f"position={position}, "
                f"subject_id={subject_id!r}, "
                f"study_id={study_id!r}, "
                f"dicom_id={dicom_id!r}"
            )

        study_key = (
            subject_id,
            study_id,
        )
        study_to_positions.setdefault(
            study_key,
            [],
        ).append(position)

    study_keys = sorted(study_to_positions)

    if subset_size > len(study_keys):
        raise ValueError(
            "请求的不同 study 数量超过当前可用 study 数量："
            f"subset_size={subset_size}, "
            f"unique_studies={len(study_keys)}"
        )

    random.Random(subset_seed).shuffle(
        study_keys
    )
    selected_studies = study_keys[
        :subset_size
    ]

    selected_positions: List[int] = []

    for subject_id, study_id in selected_studies:
        candidates = study_to_positions[
            (subject_id, study_id)
        ]

        candidates = sorted(
            candidates,
            key=lambda position: str(
                samples[position].get(
                    dicom_column,
                    "",
                )
            ),
        )

        key = (
            f"{subset_seed}\t"
            f"{subject_id}\t"
            f"{study_id}"
        )
        digest = hashlib.sha256(
            key.encode("utf-8")
        ).digest()

        selected_position = candidates[
            int.from_bytes(
                digest[:8],
                byteorder="big",
            )
            % len(candidates)
        ]

        selected_positions.append(
            selected_position
        )

    return selected_positions


def main() -> None:
    # =========================================================
    # 只需要在这里修改
    #生成时候stage1_train_config_mimic_cxr.py的代码必须是:

    """
    # v2 表示使用 mimic_cxr_text_train_cleaning
    # 过滤清洗后无效的监督文本。
    mimic_cxr_cache_prefix: str = "mimic_cxr_train_clean_v2"
    """

    # =========================================================

    # 100k 架构筛选：
    #subset_size = 100_000

    # # 80k 架构筛选：
    # subset_size = 80_000

    # 160k 架构筛选：
    subset_size = 160_000

    # 500 条 smoke test 时改成：
    # subset_size = 500

    subset_seed = 2048

    # "study_one_image" 或 "image"
    subset_strategy = "study_one_image"

    # 第一次生成可以 True；
    # 固定子集确定后建议改成 False。
    overwrite = True

    # =========================================================
    # 读取当前 Stage 1 配置和完整有效缓存
    # =========================================================
    from config.stage1_train_config_mimic_cxr import (
        TrainConfig,
    )
    from datas.mimic_cxr_datasets import (
        MIMICCXRDataset,
    )

    cfg = TrainConfig()

    # 这里必须仍指向完整 320016 缓存。
    source_cache_prefix = (
        cfg.mimic_cxr_cache_prefix
    )

    dataset = MIMICCXRDataset(
        csv_path=cfg.mimic_cxr_metadata_csv,
        image_root=cfg.mimic_cxr_image_root,
        report_root=cfg.mimic_cxr_report_root,
        target_section=cfg.mimic_cxr_target_section,
        load_report=cfg.mimic_cxr_load_report,
        allowed_view_positions=(
            cfg.mimic_cxr_view_positions
        ),
        drop_empty_target=(
            cfg.mimic_cxr_drop_empty_target
        ),
        cache_dir=cfg.mimic_cxr_cache_dir,
        use_indices_cache=True,
        rebuild_indices_cache=False,
        cache_prefix=source_cache_prefix,
    )

    source_cache_path = (
        dataset._get_indices_cache_path()
    )

    if source_cache_path is None:
        raise RuntimeError(
            "无法根据 Stage 1 配置确定一级缓存路径。"
        )

    source_indices = _load_index_list(
        source_cache_path
    )

    if len(source_indices) != len(dataset):
        raise RuntimeError(
            "一级缓存长度与加载后的有效数据集长度不一致："
            f"cache={len(source_indices)}, "
            f"dataset={len(dataset)}"
        )

    if subset_size <= 0:
        raise ValueError(
            f"subset_size 必须大于 0，当前为 {subset_size}"
        )

    if subset_size > len(dataset):
        raise ValueError(
            "subset_size 不能超过完整有效数据量："
            f"subset_size={subset_size}, "
            f"dataset_size={len(dataset)}"
        )

    if subset_strategy == "study_one_image":
        selected_positions = (
            _select_study_one_image_positions(
                dataset=dataset,
                subset_size=subset_size,
                subset_seed=subset_seed,
            )
        )
    elif subset_strategy == "image":
        selected_positions = (
            _select_image_positions(
                dataset=dataset,
                subset_size=subset_size,
                subset_seed=subset_seed,
            )
        )
    else:
        raise ValueError(
            "subset_strategy 必须是 "
            "'study_one_image' 或 'image'，"
            f"当前为 {subset_strategy!r}"
        )

    if len(selected_positions) != subset_size:
        raise RuntimeError(
            "抽样数量异常："
            f"selected={len(selected_positions)}, "
            f"expected={subset_size}"
        )

    if len(selected_positions) != len(
        set(selected_positions)
    ):
        raise RuntimeError(
            "抽样结果中存在重复的局部位置。"
        )

    # 关键：把 320016 有效数据中的局部位置，
    # 映射回原始 filtered metadata 索引。
    selected_original_indices = [
        source_indices[position]
        for position in selected_positions
    ]

    if len(selected_original_indices) != len(
        set(selected_original_indices)
    ):
        raise RuntimeError(
            "映射后的原始索引中存在重复值。"
        )

    target_cache_prefix = (
        f"{source_cache_prefix}_"
        f"screen{_size_label(subset_size)}_"
        f"seed{subset_seed}"
    )

    original_prefix = dataset.cache_prefix
    dataset.cache_prefix = (
        target_cache_prefix
    )
    target_cache_path = (
        dataset._get_indices_cache_path()
    )
    dataset.cache_prefix = original_prefix

    if target_cache_path is None:
        raise RuntimeError(
            "无法确定目标子集缓存路径。"
        )

    print("\n" + "=" * 70)
    print("MIMIC-CXR 同构 valid-indices 子集缓存")
    print("=" * 70)
    print(f"源缓存：{source_cache_path}")
    print(f"完整有效样本数：{len(dataset)}")
    print(f"目标子集大小：{subset_size}")
    print(f"抽样 seed：{subset_seed}")
    print(f"抽样策略：{subset_strategy}")
    print(f"目标 cache_prefix：{target_cache_prefix}")
    print(f"目标缓存：{target_cache_path}")
    print(f"覆盖已有文件：{overwrite}")
    print("=" * 70)

    if (
        target_cache_path.exists()
        and not overwrite
    ):
        existing_indices = _load_index_list(
            target_cache_path
        )

        if existing_indices != selected_original_indices:
            raise RuntimeError(
                "目标缓存已存在，但内容与当前固定抽样不一致。"
                "需要主动重建时再将 overwrite=True。"
            )

        print(
            "[MIMIC-CXR subset] "
            "已有同构缓存校验通过。"
        )
    else:
        _save_index_list(
            target_cache_path,
            selected_original_indices,
        )

        print(
            "[MIMIC-CXR subset] "
            "同构缓存生成完成。"
        )

    print(
        f"  selected_size="
        f"{len(selected_original_indices)}"
    )
    print(
        "  selected_sha256="
        f"{_sha256_indices(selected_original_indices)}"
    )
    print("\nStage 1 配置只需修改：")
    print(
        "mimic_cxr_cache_prefix = "
        f"{target_cache_prefix!r}"
    )
    print(
        "mimic_cxr_rebuild_indices_cache = False"
    )
    print(
        "max_mimic_cxr_train_samples = None"
    )


if __name__ == "__main__":
    main()

"""
/home/yuqing/miniconda3/envs/qwen3vl/bin/python /home/yuqing/RemoteProjects/MoRA_Med/other/generate_mimic_cxr_cache_subset_same_format.py 
当前输出目录为: /home/yuqing/Models/MoRA_Med/Stage1_MIMIC_CXR_A5_Experts-F1-F3-F5_Scale-learned_Gate-learned_Lambda-learnable_RMS-1_Seed-2048
[MIMICCXRDataset] Loaded cached valid indices: /home/yuqing/Datas/mimic-cxr-jpg-2.1.0/cache/mimic_cxr_train_clean_v2_impression_all_views_drop_empty_valid_indices.json
[MIMICCXRDataset] Samples after cache filtering: 320016

======================================================================
MIMIC-CXR 同构 valid-indices 子集缓存
======================================================================
源缓存：/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/cache/mimic_cxr_train_clean_v2_impression_all_views_drop_empty_valid_indices.json
完整有效样本数：320016
目标子集大小：100000
抽样 seed：2048
抽样策略：study_one_image
目标 cache_prefix：mimic_cxr_train_clean_v2_screen100k_seed2048
目标缓存：/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/cache/mimic_cxr_train_clean_v2_screen100k_seed2048_impression_all_views_drop_empty_valid_indices.json
覆盖已有文件：True
======================================================================
[MIMIC-CXR subset] 同构缓存生成完成。
  selected_size=100000
  selected_sha256=a2fe3d1c3d908e37e5196f472d89910c91e55a5325cad433da8ed609e1551540

Stage 1 配置只需修改：
mimic_cxr_cache_prefix = 'mimic_cxr_train_clean_v2_screen100k_seed2048'
mimic_cxr_rebuild_indices_cache = False
max_mimic_cxr_train_samples = None

Process finished with exit code 0
"""


"""
/home/yuqing/miniconda3/envs/qwen3vl/bin/python /home/yuqing/RemoteProjects/MoRA_Med/other/generate_mimic_cxr_cache_subset_same_format.py 
当前输出目录为: /home/yuqing/Models/MoRA_Med/Stage1_MIMIC_CXR_A5_Experts-F1-F3-F5_Scale-learned_Gate-learned_Lambda-learnable_RMS-1_Seed-2048
[MIMICCXRDataset] Loaded cached valid indices: /home/yuqing/Datas/mimic-cxr-jpg-2.1.0/cache/mimic_cxr_train_clean_v2_impression_all_views_drop_empty_valid_indices.json
[MIMICCXRDataset] Samples after cache filtering: 320016

======================================================================
MIMIC-CXR 同构 valid-indices 子集缓存
======================================================================
源缓存：/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/cache/mimic_cxr_train_clean_v2_impression_all_views_drop_empty_valid_indices.json
完整有效样本数：320016
目标子集大小：80000
抽样 seed：2048
抽样策略：study_one_image
目标 cache_prefix：mimic_cxr_train_clean_v2_screen80k_seed2048
目标缓存：/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/cache/mimic_cxr_train_clean_v2_screen80k_seed2048_impression_all_views_drop_empty_valid_indices.json
覆盖已有文件：True
======================================================================
[MIMIC-CXR subset] 同构缓存生成完成。
  selected_size=80000
  selected_sha256=c1793bcc0640d30cf725ee42d7daf56b0d1fffa2d4374693c9a7ff468a755068

Stage 1 配置只需修改：
mimic_cxr_cache_prefix = 'mimic_cxr_train_clean_v2_screen80k_seed2048'
mimic_cxr_rebuild_indices_cache = False
max_mimic_cxr_train_samples = None
"""