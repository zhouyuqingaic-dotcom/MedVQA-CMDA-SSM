import re


def vqa_med_2019_answer_train_cleaning(text: str) -> str:
    """
    第一层：VQA-MED-2019 训练前极轻量清洗 (极其保守)

    逻辑与 VQA-RAD / SLAKE 保持严格对齐，确保跨域泛化评测的公平性。
    特别注意：VQA-MED 中有 "cta - ct angiography" 或 "mass#neoplasm"
    这类带有内部符号的极其规范的答案，因此我们绝不能粗暴地使用正则替换掉所有符号，
    只能清理末尾的无意义标点。
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. 去首尾空白、换行、tab
    text = text.strip()

    # 2. 把连续空白折叠成单空格
    text = re.sub(r'\s+', ' ', text)

    # 3. 去掉末尾单个或连续的纯格式性标点 (. ; : ,)
    # 这样可以安全地清理掉句末句号，但绝不会破坏单词内部的连字符(-)或井号(#)
    text = re.sub(r'[.;:,]+$', '', text)

    # 再次 strip 防止去掉标点后暴露出新的末尾空格
    text = text.strip()

    # 4. 统一 yes/no 的大小写 (仅对精确等于 yes/no 的情况生效)
    lower_text = text.lower()
    if lower_text == 'yes':
        return 'yes'
    elif lower_text == 'no':
        return 'no'

    # 5. 返回极轻量清洗后的结果
    return text


def vqa_med_2019_answer_eval_cleaning(text: str) -> str:
    """
    第二层：VQA-MED-2019 评测时的 normalized 版本
    在第一层(训练清洗)的基础上，增加全局转小写，用于计算 Normalized Strict Accuracy。
    """
    # 1. 先进行基础的训练层结构化清洗
    cleaned_text = vqa_med_2019_answer_train_cleaning(text)

    # 2. 评测对账时，为避免大小写导致的严格匹配失败，全局转小写
    cleaned_text = cleaned_text.lower()

    # 3. 再次去首尾空白兜底
    cleaned_text = cleaned_text.strip()

    return cleaned_text