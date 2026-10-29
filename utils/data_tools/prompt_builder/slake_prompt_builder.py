def build_slake_prompt(question: str, instruction_suffix: str) -> str:
    """
    统一构造 SLAKE 任务的文本 Prompt (阶段二短答/实体识别专用)。

    职责：
    单纯地将清理后的问题文本与 Stage 2 专属的 instruction_suffix 拼接。
    不处理任何多选项逻辑。

    注：
    SLAKE 数据集的答案具有极强的“实体中心化 (Entity-centric)”特征
    (例如: "MRI", "Abdomen", "Yes", "Right")。
    通过与 VQA-RAD 一致的 instruction_suffix 拼接，
    引导 LLM Decoder 生成精简、精准的医学名词或判断词。
    """
    # 确保问题文本干净，去除多余的换行或首尾空格
    question_clean = question.strip()

    # 拼接格式：
    # [问题内容]
    # [指令后缀]
    prompt = f"{question_clean}\n{instruction_suffix}"

    return prompt