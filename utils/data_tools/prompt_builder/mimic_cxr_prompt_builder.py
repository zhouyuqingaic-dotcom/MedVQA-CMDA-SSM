def build_mimic_cxr_prompt(instruction: str = "Describe the radiographic findings of this image briefly.") -> str:
    """
    统一构造 MIMIC-CXR 任务的纯文本 Prompt。

    由于 MIMIC-CXR 本质上是一个 Captioning / Report Generation 任务，
    通常不需要像 VQA 那样传入具体的 question，只需要一个全局的固定指令即可。
    """
    # 清理头尾多余空白
    prompt = instruction.strip()

    return prompt