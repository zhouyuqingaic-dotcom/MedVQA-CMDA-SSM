import re

def slake_answer_train_cleaning(text: str) -> str:
    """
    第一层：SLAKE 训练前极轻量清洗
    逻辑与 VQA-RAD 保持严格对齐，确保跨域泛化评测的公平性。
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. 去首尾空白、换行、tab
    text = text.strip()

    # 2. 把连续空白折叠成单空格
    text = re.sub(r'\s+', ' ', text)

    # 3. 去掉末尾单个或连续的格式性标点 (. ; : ,)
    text = re.sub(r'[.;:,]+$', '', text)

    # 再次 strip 防止去掉标点后暴露出新的末尾空格
    text = text.strip()

    # 4. 统一 yes/no 的大小写
    lower_text = text.lower()
    if lower_text == 'yes':
        return 'yes'
    elif lower_text == 'no':
        return 'no'

    return text


def slake_answer_eval_cleaning(text: str) -> str:
    """
    第二层：SLAKE 评测时的 normalized 版本
    在第一层(训练清洗)的基础上，增加全局转小写，用于计算 Normalized Accuracy。
    """
    # 1. 先进行基础的训练层清洗
    cleaned_text = slake_answer_train_cleaning(text)

    # 2. 评测时全局转小写
    cleaned_text = cleaned_text.lower()

    # 3. 再次去首尾空白兜底
    cleaned_text = cleaned_text.strip()

    return cleaned_text