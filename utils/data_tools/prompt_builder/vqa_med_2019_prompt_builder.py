def build_vqa_med_2019_prompt(question: str, instruction_suffix: str) -> str:
    """
    统一构造 VQA-MED-2019 任务的文本 Prompt (阶段二短答/分类专用)。

    职责：
    单纯地将清理后的问题文本与 Stage 2 专属的 instruction_suffix 拼接。
    不处理任何多选项逻辑。

    注：
    VQA-MED-2019 包含四种类型的问题 (Modality, Plane, Organ, Abnormality)。
    通过与统一的 instruction_suffix 拼接，
    引导 LLM Decoder 稳定输出符合这四个维度的短文本或 Yes/No。
    """
    # 确保问题文本干净，没有多余的换行或首尾空格
    question_clean = question.strip()

    # 拼接格式：
    # [问题内容]
    # [指令后缀]
    prompt = f"{question_clean}\n{instruction_suffix}"

    return prompt