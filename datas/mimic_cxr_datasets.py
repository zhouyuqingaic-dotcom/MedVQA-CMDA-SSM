import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.data import Dataset

from utils.data_tools.prompt_cleaning.mimic_cxr_text_cleaning import (
    mimic_cxr_text_train_cleaning,
)


class MIMICCXRDataset(Dataset):
    """
    精简版 MIMIC-CXR-JPG Dataset

    设计目标
    ----------
    1. 支持读取 metadata.csv(.gz) / 自定义 QA csv
    2. 不依赖 split.csv，不做 split merge，不做 split 过滤
    3. 自动兼容 reports/ 与 reports/files/ 两种报告根目录
    4. 稳定抽取 findings / impression / full_report
    5. 输出字段统一，方便直接接训练脚本

    常见用法
    ----------
    1) 图像 -> Impression
       csv_path=metadata.csv.gz
       target_section="impression"

    2) 图像 -> Findings
       csv_path=metadata.csv.gz
       target_section="findings"

    3) 图像 + 问题 -> 答案
       csv_path=你的QA表
       question_column="question"
       answer_column="answer"
    """

    VALID_TARGET_SECTIONS = {"impression", "findings", "full_report"}

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        report_root: Optional[str] = None,
        image_path_column: Optional[str] = None,
        question_column: Optional[str] = None,
        answer_column: Optional[str] = None,
        subject_id_column: str = "subject_id",
        study_id_column: str = "study_id",
        dicom_id_column: str = "dicom_id",
        view_position_column: Optional[str] = "ViewPosition",
        target_section: str = "impression",
        load_report: bool = True,
        allowed_view_positions: Optional[Sequence[str]] = None,
        drop_empty_target: bool = False,
        cache_dir: Optional[str] = None,
        use_indices_cache: bool = True,
        rebuild_indices_cache: bool = False,
        cache_prefix: str = "mimic_cxr",
        encoding: str = "utf-8",
    ) -> None:
        #设置缓存
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_indices_cache = use_indices_cache
        self.rebuild_indices_cache = rebuild_indices_cache
        self.cache_prefix = cache_prefix


        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.report_root = Path(report_root) if report_root is not None else None

        self.image_path_column = image_path_column
        self.question_column = question_column
        self.answer_column = answer_column

        self.subject_id_column = subject_id_column
        self.study_id_column = study_id_column
        self.dicom_id_column = dicom_id_column
        self.view_position_column = view_position_column

        self.target_section = target_section
        self.load_report = load_report
        self.allowed_view_positions = (
            tuple(allowed_view_positions) if allowed_view_positions is not None else None
        )
        self.drop_empty_target = drop_empty_target
        self.encoding = encoding

        self._validate_init_args()

        self.samples: List[Dict[str, str]] = self._read_csv(self.csv_path)
        self._validate_required_columns()
        self._filter_rows()



    # ------------------------------------------------------------------
    # 初始化校验
    # ------------------------------------------------------------------
    def _validate_init_args(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {self.csv_path}")

        if not self.image_root.exists():
            raise FileNotFoundError(f"图像根目录不存在: {self.image_root}")

        if self.load_report:
            if self.report_root is None:
                raise ValueError("当 load_report=True 时，必须传入 report_root。")
            if not self.report_root.exists():
                raise FileNotFoundError(f"报告根目录不存在: {self.report_root}")

        if self.target_section not in self.VALID_TARGET_SECTIONS:
            raise ValueError(
                f"target_section 只能是 {sorted(self.VALID_TARGET_SECTIONS)}，当前为: {self.target_section}"
            )

    # ------------------------------------------------------------------
    # 读 CSV
    # ------------------------------------------------------------------
    def _open_text_file(self, path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding=self.encoding, newline="")
        return open(path, "r", encoding=self.encoding, newline="")

    def _read_csv(self, path: Path) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []

        with self._open_text_file(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row: Dict[str, str] = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    clean_key = str(key).strip()
                    clean_row[clean_key] = "" if value is None else str(value).strip()
                rows.append(clean_row)

        if not rows:
            raise ValueError(f"CSV 文件为空或没有有效样本: {path}")

        return rows

    # ------------------------------------------------------------------
    # 列校验
    # ------------------------------------------------------------------
    def _validate_required_columns(self) -> None:
        existing_columns = set(self.samples[0].keys())
        required_columns: List[str] = []

        if self.image_path_column is not None:
            required_columns.append(self.image_path_column)
        else:
            required_columns.extend([
                self.subject_id_column,
                self.study_id_column,
                self.dicom_id_column,
            ])

        if self.question_column is not None:
            required_columns.append(self.question_column)

        if self.answer_column is not None:
            required_columns.append(self.answer_column)

        missing_columns = [col for col in required_columns if col not in existing_columns]
        if missing_columns:
            raise ValueError(
                f"CSV 缺少必要列: {missing_columns}\n"
                f"当前 CSV: {self.csv_path}\n"
                f"已有列: {sorted(existing_columns)}"
            )

    # ------------------------------------------------------------------
    # 行过滤
    # ------------------------------------------------------------------
    def _filter_rows(self) -> None:
        filtered = self.samples

        # 先做 view 过滤，这部分很快
        if self.allowed_view_positions is not None and self.view_position_column is not None:
            allowed = {v.strip().upper() for v in self.allowed_view_positions}
            filtered = [
                row for row in filtered
                if row.get(self.view_position_column, "").strip().upper() in allowed
            ]

        # 不需要按 target 过滤，直接结束
        if not (self.drop_empty_target and self.load_report):
            self.samples = filtered
            return

        # 下面是“空 target 过滤”，优先走缓存
        cache_path = self._get_indices_cache_path()

        if (
                self.use_indices_cache
                and not self.rebuild_indices_cache
                and cache_path is not None
        ):
            cached_indices = self._load_cached_indices(cache_path)
            if cached_indices is not None:
                self.samples = [
                    filtered[i] for i in cached_indices
                    if 0 <= i < len(filtered)
                ]
                print(f"[MIMICCXRDataset] Loaded cached valid indices: {cache_path}")
                print(f"[MIMICCXRDataset] Samples after cache filtering: {len(self.samples)}")
                return

        # 没有缓存，或者强制重建，就慢扫一次。
        kept_rows: List[Dict[str, str]] = []
        kept_indices: List[int] = []

        raw_empty_count = 0
        cleaned_empty_count = 0

        for i, row in enumerate(filtered):
            report_path = self._build_report_path(row)
            report_text = self._read_report(report_path)
            parsed = self._parse_report_sections(report_text)

            raw_target_text = parsed["target_text"]

            if not raw_target_text.strip():
                raw_empty_count += 1
                continue

            cleaned_target_text = mimic_cxr_text_train_cleaning(
                raw_target_text
            )

            if not cleaned_target_text:
                cleaned_empty_count += 1
                continue

            kept_rows.append(row)
            kept_indices.append(i)

        self.samples = kept_rows

        print(
            "[MIMICCXRDataset] Filter statistics: "
            f"input={len(filtered)}, "
            f"raw_empty={raw_empty_count}, "
            f"cleaned_empty={cleaned_empty_count}, "
            f"kept={len(kept_rows)}"
        )

        if self.use_indices_cache and cache_path is not None:
            self._save_cached_indices(
                cache_path,
                kept_indices,
            )
            print(
                "[MIMICCXRDataset] "
                f"Saved valid indices cache: {cache_path}"
            )

        print(
            "[MIMICCXRDataset] "
            f"Samples after rebuilding filter: {len(self.samples)}"
        )


    # ------------------------------------------------------------------
    # 路径构建
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_id(value: str) -> str:
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return text

    def _build_image_path(self, row: Dict[str, str]) -> Path:
        if self.image_path_column is not None:
            raw_path = row[self.image_path_column]
            path_obj = Path(raw_path)
            return path_obj if path_obj.is_absolute() else self.image_root / path_obj

        subject_id = self._normalize_id(row[self.subject_id_column])
        study_id = self._normalize_id(row[self.study_id_column])
        dicom_id = row[self.dicom_id_column].strip()

        prefix = f"p{subject_id[:2]}"
        return self.image_root / prefix / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"

    def _build_report_path(self, row: Dict[str, str]) -> Optional[Path]:
        if self.report_root is None:
            return None

        subject_id = self._normalize_id(row.get(self.subject_id_column, ""))
        study_id = self._normalize_id(row.get(self.study_id_column, ""))

        if not subject_id or not study_id:
            return None

        prefix = f"p{subject_id[:2]}"

        candidate_1 = self.report_root / prefix / f"p{subject_id}" / f"s{study_id}.txt"
        candidate_2 = self.report_root / "files" / prefix / f"p{subject_id}" / f"s{study_id}.txt"

        if candidate_1.exists():
            return candidate_1
        if candidate_2.exists():
            return candidate_2

        if (self.report_root / "files").exists():
            return candidate_2
        return candidate_1

    # ------------------------------------------------------------------
    # 报告读取与解析
    # ------------------------------------------------------------------
    def _read_report(self, report_path: Optional[Path]) -> str:
        if report_path is None or not report_path.exists():
            return ""

        with open(report_path, "r", encoding=self.encoding) as f:
            return f.read().strip()

    @staticmethod
    def _normalize_report_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_section(report_text: str, section_name: str) -> str:
        """
        更稳的 section 提取：
        - 支持 FINDINGS: / IMPRESSION:
        - 到下一个大写标题结束
        - 避免把 IMPRESSION 混进 FINDINGS
        """
        if not report_text.strip():
            return ""

        text = MIMICCXRDataset._normalize_report_text(report_text)

        pattern = (
            rf"(?ims)^\s*{re.escape(section_name)}\s*:?\s*(.*?)"
            rf"(?=^\s*[A-Z][A-Z /_-]{{2,}}\s*:?\s*$|^\s*[A-Z][A-Z /_-]{{2,}}\s*:|\Z)"
        )
        match = re.search(pattern, text)
        if not match:
            return ""

        section_text = match.group(1).strip()

        section_text = re.split(
            r"(?i)\b(?:FINDINGS|IMPRESSION|HISTORY|INDICATION|COMPARISON|EXAMINATION|TECHNIQUE)\s*:",
            section_text,
        )[0].strip()

        section_text = re.sub(r"\s+", " ", section_text)
        return section_text

    def _parse_report_sections(self, report_text: str) -> Dict[str, str]:
        if not report_text.strip():
            return {
                "report": "",
                "findings": "",
                "impression": "",
                "target_text": "",
            }

        text = self._normalize_report_text(report_text)
        findings = self._extract_section(text, "FINDINGS")
        impression = self._extract_section(text, "IMPRESSION")

        if self.target_section == "findings":
            target_text = findings
        elif self.target_section == "impression":
            target_text = impression
        else:
            target_text = re.sub(r"\s+", " ", text).strip()

        return {
            "report": text,
            "findings": findings,
            "impression": impression,
            "target_text": target_text,
        }

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.samples[index]

        image_path = self._build_image_path(row)
        report_path = self._build_report_path(row)

        report_text = self._read_report(report_path) if self.load_report else ""
        parsed = self._parse_report_sections(report_text)

        sample: Dict[str, Any] = {
            "index": index,
            "image_path": str(image_path),
            "report_path": str(report_path) if report_path is not None else None,
            "subject_id": row.get(self.subject_id_column, ""),
            "study_id": row.get(self.study_id_column, ""),
            "dicom_id": row.get(self.dicom_id_column, ""),
            "question": row.get(self.question_column, "") if self.question_column is not None else None,
            "answer": row.get(self.answer_column, "") if self.answer_column is not None else None,
            "report": parsed["report"],
            "findings": parsed["findings"],
            "impression": parsed["impression"],
            "target_text": parsed["target_text"],
            "view_position": row.get(self.view_position_column, "") if self.view_position_column is not None else "",
            "raw_row": row,
        }

        return sample

    def _get_indices_cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        section = self.target_section.lower()
        if self.allowed_view_positions:
            views = "_".join(sorted(v.lower() for v in self.allowed_view_positions))
        else:
            views = "all_views"

        empty_flag = "drop_empty" if self.drop_empty_target else "keep_empty"

        filename = f"{self.cache_prefix}_{section}_{views}_{empty_flag}_valid_indices.json"
        return self.cache_dir / filename

    def _load_cached_indices(self, cache_path: Path) -> Optional[List[int]]:
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                return None

            if not all(isinstance(x, int) for x in data):
                return None

            return data
        except Exception:
            return None

    def _save_cached_indices(self, cache_path: Path, indices: List[int]) -> None:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(indices, f, ensure_ascii=False)

