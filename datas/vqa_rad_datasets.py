import json
from pathlib import Path
from typing import Any, Dict, List
from torch.utils.data import Dataset


class VQARADDataset(Dataset):
    """
    VQA-RAD 纯净版数据集读取类。

    职责：
    - 读取 download_vqa_rad.py 生成的 JSONL 文件。
    - 拼接并验证图像路径。
    - 向下层 Collator 吐出原生数据字典。
    """

    def __init__(
            self,
            jsonl_path: str,
            image_root: str,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL 数据文件不存在: {self.jsonl_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"图像根目录不存在: {self.image_root}")

        self.samples: List[Dict[str, Any]] = self._read_jsonl(self.jsonl_path)

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        向 Collator 提供单条样本。

        返回结构：
        - index: 整数索引
        - image_path: 拼接好的本地绝对路径
        - question: 提问字符串
        - answer: 真实短答案字符串
        - question_type: 题目类型 (原生自带, 评测切片分析用)
        - answer_type: 答案类型 (原生自带, closed/open)
        - raw_row: 原始字典，兜底用
        """
        row = self.samples[index]
        image_name = row["image_name"]
        image_path = self.image_root / image_name

        # 强转字符串，防止原生数据里有非 str 类型的 answer (比如布尔值或整数)
        answer_text = str(row.get("answer", "")).strip()

        sample = {
            "index": index,
            "image_path": str(image_path),
            "question": row.get("question", ""),
            "answer": answer_text,
            "question_type": row.get("question_type", ""),
            "answer_type": row.get("answer_type", ""),
            "raw_row": row
        }

        return sample