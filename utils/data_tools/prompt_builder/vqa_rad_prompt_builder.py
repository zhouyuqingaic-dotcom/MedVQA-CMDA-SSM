def build_vqa_rad_prompt(question: str, instruction_suffix: str) -> str:
    """
    统一构造 VQA-RAD 任务的文本 Prompt (阶段二开放式短答专用)。

    职责：
    单纯地将清理后的问题文本与 Stage 2 专属的 instruction_suffix 拼接。
    不处理任何多选项逻辑。
    """
    # 确保问题文本干净，没有多余的换行
    question_clean = question.strip()

    # 拼接格式：
    # [问题内容]
    # [指令后缀]
    prompt = f"{question_clean}\n{instruction_suffix}"

    return prompt