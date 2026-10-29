import re


def mimic_cxr_text_train_cleaning(text: str) -> str:
    """
    MIMIC-CXR 专属的报告文本清洗器 (段落/篇章级别)。
    目标：去除训练噪声，规范化排版，过滤无效监督信号。
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. 统一去标识化占位符
    # MIMIC-CXR 官方规范是 '___'，但早期记录可能混杂 '[**...**]'。这里统一转为 '___'
    text = re.sub(r'\[\*\*.*?\*\*\]', '___', text)

    # 2. 去掉明显残留的 Section 前缀 (防漏网之鱼)
    # 【修复点】：在 Python 3.11+ 中，内联 flag (?i) 必须放在正则表达式的最开头！
    pattern_section = r'(?i)^(impression|findings|history|indication|conclusion|report)\s*:\s*'
    text = re.sub(pattern_section, '', text).strip()

    # 3. 标点和多余空白规范化
    # 3.1 折叠多余换行：最多保留两个换行（允许保留正常的段落分割）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 3.2 折叠连续空格和 Tab 为单空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 3.3 修复标点前多余的空格："mild edema ." -> "mild edema."
    text = re.sub(r'\s+([.;:,])', r'\1', text)

    # 4. 清理首尾残留的空白和无意义的下划线
    text = text.strip()

    # 5. 过滤无效 Supervision
    # 如果清洗完只剩这些毫无意义的“废话”，直接截断为空字符串
    lower_text = text.lower()
    invalid_supervisions = {"", "none", "n/a", "none.", "n/a.", ".", "...", "___", "___."}

    if lower_text in invalid_supervisions:
        return ""

    # (可选) 6. 保证末尾句号存在
    # 放射学报告中，医生经常漏打最后一个句号。帮模型规范化习惯。
    if text and text[-1] not in ['.', '!', '?', ';', ':']:
        text += '.'

    return text