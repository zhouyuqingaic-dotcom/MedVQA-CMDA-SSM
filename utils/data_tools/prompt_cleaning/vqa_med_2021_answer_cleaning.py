from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


# 仅在评测归一化时统一常见 Unicode 字符。
#
# VQA-Med 2021 官方答案中可能出现：
#   hodgkin’s lymphoma
# 而模型可能生成：
#   hodgkin's lymphoma
#
# 两者医学含义相同，因此评测时统一为 ASCII 形式。
_EVAL_TRANSLATION_TABLE = str.maketrans(
    {
        "\u2018": "'",   # left single quotation mark
        "\u2019": "'",   # right single quotation mark
        "\u201b": "'",   # single high-reversed-9 quotation mark
        "\u02bc": "'",   # modifier letter apostrophe
        "\u2032": "'",   # prime
        "\u201c": '"',   # left double quotation mark
        "\u201d": '"',   # right double quotation mark
        "\u201e": '"',   # double low-9 quotation mark
        "\u2010": "-",   # hyphen
        "\u2011": "-",   # non-breaking hyphen
        "\u2012": "-",   # figure dash
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2212": "-",   # minus sign
        "\u00a0": " ",   # non-breaking space
    }
)


def vqa_med_2021_answer_train_cleaning(text: str) -> str:
    """
    VQA-Med 2021 训练答案的极轻量、保守清洗。

    设计原则
    --------
    与 SLAKE、VQA-RAD、VQA-Med 2019 的现有清洗策略保持一致：
        1. 去除首尾空白；
        2. 折叠连续空白；
        3. 仅去除答案末尾的格式性标点；
        4. 仅对精确的 yes/no 统一为小写；
        5. 不破坏医学术语内部结构。

    特别保留
    --------
    不会删除或重写以下可能携带医学语义的内容：
        - 连字符：non-aggressive、chance-type
        - 括号：systemic lupus erythematosus (sle)
        - 数字和分级：c5 c6、t1、nf-1
        - 内部逗号：carcinoma, small cell
        - 内部句号：较长影像学描述中的完整句子
        - 左右侧和解剖位置
        - Unicode 医学文本字符

    Parameters
    ----------
    text:
        原始训练答案。

    Returns
    -------
    str
        清洗后的训练目标。非法类型或空文本返回空字符串。
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. 去除首尾空格、换行和 tab。
    cleaned = text.strip()

    # 2. 连续空白折叠为一个普通空格。
    cleaned = re.sub(r"\s+", " ", cleaned)

    # 3. 只删除末尾的格式性标点。
    #
    # 例如：
    #   "pneumonia." -> "pneumonia"
    #
    # 但不会破坏：
    #   "carcinoma, small cell"
    #   "neurofibromatosis-1, nf1, nf-1"
    #   "The spinal canal is diffusely remodeled and enlarged."
    #
    # 对最后一个例子，只会删除最末尾句号，内部句号保持不变。
    cleaned = re.sub(r"[.;:,]+$", "", cleaned).strip()

    # 4. 仅对完整答案恰好为 yes/no 时统一大小写。
    lowered = cleaned.lower()

    if lowered == "yes":
        return "yes"

    if lowered == "no":
        return "no"

    return cleaned


def vqa_med_2021_answer_eval_cleaning(text: str) -> str:
    """
    VQA-Med 2021 单个答案的评测归一化。

    在训练清洗基础上额外执行：
        1. 常见 Unicode 引号、破折号和不间断空格归一化；
        2. 再次折叠空白；
        3. 全局转为小写。

    该函数用于：
        - 模型预测归一化；
        - 主参考答案归一化；
        - 多参考答案逐条归一化；
        - Normalized Exact Match 计算。

    该函数仍然保持保守，不会：
        - 删除全部标点；
        - 删除冠词；
        - 词干化；
        - 同义词替换；
        - 改写医学诊断；
        - 将长答案截断为第一个短语。

    Parameters
    ----------
    text:
        模型预测或官方参考答案。

    Returns
    -------
    str
        用于确定性评测的 normalized answer。
    """
    cleaned = vqa_med_2021_answer_train_cleaning(text)

    if not cleaned:
        return ""

    cleaned = cleaned.translate(
        _EVAL_TRANSLATION_TABLE
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    return cleaned.lower()


def vqa_med_2021_references_eval_cleaning(
    references: Sequence[Any] | str | None,
) -> list[str]:
    """
    对 VQA-Med 2021 的多参考答案进行评测归一化。

    Test JSONL 中可能包含：

        "references": [
            "avascular necrosis",
            "extensive degenerative changes ...",
            "Avascular Necrosis of femoral heads ..."
        ]

    本函数逐条调用 vqa_med_2021_answer_eval_cleaning，并：
        - 删除清洗后为空的参考答案；
        - 按原顺序去重；
        - 保留所有不同的有效参考表达。

    Parameters
    ----------
    references:
        单个字符串、字符串序列或 None。

    Returns
    -------
    list[str]
        清洗和去重后的参考答案列表。

    Raises
    ------
    TypeError
        references 既不是字符串，也不是序列或 None。
    """
    if references is None:
        return []

    if isinstance(references, str):
        raw_references: Sequence[Any] = [references]
    elif isinstance(references, Sequence):
        raw_references = references
    else:
        raise TypeError(
            "VQA-Med 2021 references 必须是 str、Sequence 或 None，"
            f"当前类型为 {type(references)}。"
        )

    normalized_references: list[str] = []
    seen: set[str] = set()

    for reference in raw_references:
        # Dataset 层已保证 references 为字符串列表。
        # 这里仍做容错：None 被视为空，其余值转字符串。
        if reference is None:
            continue

        if isinstance(reference, str):
            reference_text = reference
        else:
            reference_text = str(reference)

        normalized = (
            vqa_med_2021_answer_eval_cleaning(
                reference_text
            )
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        normalized_references.append(
            normalized
        )

    return normalized_references
