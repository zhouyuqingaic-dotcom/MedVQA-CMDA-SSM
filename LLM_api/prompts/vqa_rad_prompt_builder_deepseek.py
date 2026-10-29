import json


def build_llm_judge_user_prompt(question: str, gt_raw: str, gt_norm: str, pred_raw: str, pred_norm: str) -> str:
    """
    构造传给 deepseek 裁判的 User Prompt。
    同时传入原始文本(Raw)和规范化文本(Normalized)，让 LLM 有充分的上下文进行裁判。
    """
    user_prompt = (
        f"Question: {str(question).strip()}\n"
        f"Raw Ground Truth: {str(gt_raw).strip()}\n"
        f"Normalized Ground Truth: {str(gt_norm).strip()}\n"
        f"Raw Prediction: {str(pred_raw).strip()}\n"
        f"Normalized Prediction: {str(pred_norm).strip()}\n\n"
        "Please evaluate the Prediction against the Ground Truth based on the criteria, "
        "and output the raw JSON directly."
    )
    return user_prompt


def parse_llm_judge_response(response_text: str) -> dict:
    """
    安全解析大模型返回的 JSON 字符串。
    应对大模型带有 ```json 外壳的情况，并验证字段正确性。
    """
    if response_text == "ERROR":
        return {"score": "incorrect", "reasoning": "API Request Failed."}

    text = response_text.strip()

    # 剥离 markdown 外壳
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        result = json.loads(text)
        # 确保返回格式符合预期，兜底防崩溃
        if "score" not in result or result["score"] not in ["correct", "partially_correct", "incorrect"]:
            result["score"] = "incorrect"
            result["reasoning"] = f"Invalid Score generated. Raw Output: {text}"

        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided by model."

        return result
    except json.JSONDecodeError:
        return {
            "score": "incorrect",
            "reasoning": f"JSON Decode Error. Raw Output: {text}"
        }