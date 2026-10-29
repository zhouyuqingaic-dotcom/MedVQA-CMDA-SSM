from __future__ import annotations


def build_vqa_med_2021_prompt(
    question: str,
    instruction_suffix: str,
) -> str:
    """
    构造 VQA-Med 2021 Task 1 的 Qwen3-VL 文本 Prompt。

    本函数只负责：
        原始医学问题 + Stage 2 回答格式指令

    本函数不负责：
        - 图像加载或预处理
        - Chat Template 构造
        - Tokenize
        - 答案清洗
        - 多参考答案处理
        - BioMedCLIP Router 文本构造

    说明
    ----
    VQA-Med 2021 Task 1 主要要求识别医学图像中的异常、
    疾病或关键影像学发现，因此应引导模型输出简短、直接的
    医学术语或短语。

    BioMedCLIP Router 仍应使用原始 question，而不是本函数
    返回的带 instruction_suffix 的完整 Prompt。这与 SLAKE、
    VQA-RAD 和 VQA-Med 2019 的现有数据流保持一致。

    Parameters
    ----------
    question:
        JSONL 中的原始医学问题。

    instruction_suffix:
        Stage 2 配置文件中的回答格式指令，例如：

        "Answer the question briefly and directly based on the image. "
        "State the primary abnormality, diagnosis, or imaging finding "
        "using a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no. "
        "Do not add unnecessary explanation."

    Returns
    -------
    str
        格式为：

        [清理后的原始问题]
        [instruction_suffix]

    Raises
    ------
    TypeError
        question 或 instruction_suffix 不是字符串。

    ValueError
        question 或 instruction_suffix 清理后为空。
    """
    if not isinstance(question, str):
        raise TypeError(
            "VQA-Med 2021 question 必须是 str，"
            f"当前类型为 {type(question)}。"
        )

    if not isinstance(instruction_suffix, str):
        raise TypeError(
            "VQA-Med 2021 instruction_suffix 必须是 str，"
            f"当前类型为 {type(instruction_suffix)}。"
        )

    # 使用 split/join 折叠空格、换行和 tab，避免脏文本破坏 Prompt 格式。
    question_clean = " ".join(question.split())
    instruction_clean = " ".join(instruction_suffix.split())

    if not question_clean:
        raise ValueError(
            "VQA-Med 2021 question 清理后为空。"
        )

    if not instruction_clean:
        raise ValueError(
            "VQA-Med 2021 instruction_suffix 清理后为空。"
        )

    return f"{question_clean}\n{instruction_clean}"
