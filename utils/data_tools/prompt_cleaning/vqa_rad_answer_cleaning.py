import re

def vqa_rad_answer_train_cleaning(text: str) -> str:
    """
    第一层：VQA-RAD 训练前极轻量清洗 (极其保守)
    由于 VQA-RAD 数据相对干净，这里只做最基本的表面去噪：
    1. 去首尾空白
    2. 折叠多余的连续空白为单空格
    3. 仅去掉末尾的纯格式性标点 (. ; : ,)
    4. 针对纯粹的 yes/no 变体统一转为全小写
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. 去首尾空白、换行、tab
    text = text.strip()

    # 2. 把连续空白折叠成单空格
    text = re.sub(r'\s+', ' ', text)

    # 3. 去掉末尾单个或连续的格式性标点 (. ; : ,)
    # 这样可以安全地把 "3.4 cm." 变成 "3.4 cm"，但绝不会破坏 "E. coli"
    text = re.sub(r'[.;:,]+$', '', text)

    # 再次 strip 防止去掉标点后暴露出新的末尾空格
    text = text.strip()

    # 4. 统一 yes/no 的大小写 (仅对精确等于 yes/no 的情况生效)
    # 如果是 "Yes, it is." 则不会走这个分支，完美保留原句
    lower_text = text.lower()
    if lower_text == 'yes':
        return 'yes'
    elif lower_text == 'no':
        return 'no'

    # 5. 返回极轻量清洗后的结果
    return text


def vqa_rad_answer_eval_cleaning(text: str) -> str:
    """
    第二层：VQA-RAD 评测时的 normalized 版本
    在第一层(训练清洗)的基础上，增加全局转小写，用于计算 Normalized Accuracy。
    不粗暴地去除所有标点符号，防止破坏医学语义。
    """
    # 1. 先进行基础的训练层清洗
    cleaned_text = vqa_rad_answer_train_cleaning(text)

    # 2. 评测时全局转小写
    cleaned_text = cleaned_text.lower()

    # 3. 再次去首尾空白兜底
    cleaned_text = cleaned_text.strip()

    return cleaned_text