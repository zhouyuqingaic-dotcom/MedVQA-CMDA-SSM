from pathlib import Path
from typing import Any, Dict, List
from torch.utils.data import Dataset


class VQAMED2019Dataset(Dataset):
    """
    VQA-MED-2019 纯净版数据集读取类。
    支持直接读取 QAPairsByCategory 目录，自动解析分类标签。
    兼容 Train/Val 的 3段式 和 Test 的 4段式。
    """

    def __init__(
            self,
            data_path: str,
            image_root: str,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_root = Path(image_root)

        if not self.data_path.exists():
            raise FileNotFoundError(f"VQA-MED 数据路径不存在: {self.data_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"图像根目录不存在: {self.image_root}")

        self.samples: List[Dict[str, Any]] = self._read_data(self.data_path)

    def _read_data(self, path: Path) -> List[Dict[str, Any]]:
        samples = []

        # 💡 智能路径处理：如果是目录，就遍历里面的所有 .txt；如果是文件，就只读这一个
        files_to_read = list(path.glob("*.txt")) if path.is_dir() else [path]

        for file_path in files_to_read:
            # 💡 从文件名中智能提取 Category (针对 Train / Val)
            category = "unknown"
            filename = file_path.name.lower()
            if "modality" in filename:
                category = "Modality"
            elif "plane" in filename:
                category = "Plane"
            elif "organ" in filename:
                category = "Organ"
            elif "abnormality" in filename:
                category = "Abnormality"

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("|")

                    # 💡 兼容 Test 集的 4 段式: ID | Category | Question | Answer
                    if len(parts) == 4:
                        samples.append({
                            "image_name": parts[0].strip(),
                            "category": parts[1].strip().capitalize(),  # 提取官方自带的分类并首字母大写
                            "question": parts[2].strip(),
                            "answer": parts[3].strip(),
                        })
                    # 💡 兼容 Train/Val 的 3 段式: ID | Question | Answer
                    elif len(parts) == 3:
                        samples.append({
                            "image_name": parts[0].strip(),
                            "question": parts[1].strip(),
                            "answer": parts[2].strip(),
                            "category": category  # 依赖文件名提取的分类
                        })
        return samples

    def _build_image_path(self, row: Dict[str, Any]) -> Path:
        image_name = str(row["image_name"])
        # VQA-MED 的图片后缀通常是 .jpg
        if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
            image_name += ".jpg"

        return self.image_root / image_name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.samples[index]
        image_path = self._build_image_path(row)

        answer_text = str(row.get("answer", "")).strip()

        # 💡 判断 Closed/Open
        is_closed = answer_text.lower() in ["yes", "no"]
        ans_type = "CLOSED" if is_closed else "OPEN"

        sample = {
            "index": index,
            "image_path": str(image_path),
            "question": str(row.get("question", "")).strip(),
            "answer": answer_text,
            "question_type": str(row["category"]),
            "answer_type": ans_type,
            "raw_row": row
        }
        return sample