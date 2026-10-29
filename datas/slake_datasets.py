import json
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset


class SLAKEDataset(Dataset):
    """
    SLAKE 纯净版数据集读取类。

    职责：
    - 读取 SLAKE 官方 train.json / validate.json / test.json 文件。
    - 拼接并验证图像路径。
    - 向下层 Collator 吐出原生数据字典。

    注意：
    - Dataset 只负责读数据，不负责 tokenize、图像预处理、prompt 构造。
    - 不在 Dataset 里做语言过滤；给什么 JSON 就读什么 JSON。
    """

    def __init__(
            self,
            json_path: str,
            image_root: str,
            verify_images: bool = False,
    ) -> None:
        self.json_path = Path(json_path)
        self.image_root = Path(image_root)
        self.verify_images = verify_images

        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON 数据文件不存在: {self.json_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"图像根目录不存在: {self.image_root}")

        self.samples: List[Dict[str, Any]] = self._read_json(self.json_path)

        if self.verify_images:
            self._verify_all_images()

    def _read_json(self, path: Path) -> List[Dict[str, Any]]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            samples = data
        else:
            raise ValueError(f"SLAKE 标注文件格式不正确，期望 list[dict]: {path}")

        return samples

    def _build_image_path(self, row: Dict[str, Any]) -> Path:
        """
        构造图像路径。

        SLAKE 常见字段：
        - img_name: 图像相对路径或文件名
        - img_id: 图像 ID

        优先使用 img_name；如果没有 img_name，则尝试用 img_id 构造路径。
        """
        image_name = row.get("img_name", "")

        if image_name:
            image_path = Path(str(image_name))
            if image_path.is_absolute():
                return image_path
            return self.image_root / image_path

        img_id = row.get("img_id", "")
        if img_id == "":
            raise KeyError("样本中缺少 img_name / img_id 字段，无法构造图像路径。")

        img_id = str(img_id)

        candidates = [
            self.image_root / img_id,
            self.image_root / f"{img_id}.jpg",
            self.image_root / f"{img_id}.png",
            self.image_root / f"xmlab{img_id}" / "source.jpg",
            self.image_root / f"xmlab{img_id}" / "source.png",
        ]

        for path in candidates:
            if path.exists():
                return path

        # 不强行报错，方便后续由 collator / image loader 暴露具体错误。
        return candidates[0]

    def _verify_all_images(self) -> None:
        missing = []

        for index, row in enumerate(self.samples):
            image_path = self._build_image_path(row)
            if not image_path.exists():
                missing.append((index, str(image_path)))

        if missing:
            preview = "\n".join(
                f"index={index}, image_path={image_path}"
                for index, image_path in missing[:10]
            )
            raise FileNotFoundError(
                f"发现 {len(missing)} 个图像文件不存在，前 10 个如下:\n{preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        向 Collator 提供单条样本。

        返回结构：
        - index: 整数索引
        - image_path: 拼接好的本地图像路径
        - question: 提问字符串
        - answer: 真实短答案字符串
        - question_type: 题目类型 / 内容类型
        - answer_type: 答案类型，open / closed
        - image_id: 图像 ID
        - image_name: 图像文件名或相对路径
        - raw_row: 原始字典，兜底用
        """
        row = self.samples[index]

        image_path = self._build_image_path(row)

        # 强转字符串，防止原生数据里有非 str 类型。
        question_text = str(row.get("question", "")).strip()
        answer_text = str(row.get("answer", "")).strip()

        sample = {
            "index": index,
            "image_path": str(image_path),
            "question": question_text,
            "answer": answer_text,
            "question_type": row.get("content_type", row.get("question_type", "")),
            "answer_type": row.get("answer_type", ""),
            "image_id": str(row.get("img_id", "")),
            "image_name": str(row.get("img_name", "")),
            "raw_row": row,
        }

        return sample