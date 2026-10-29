import os
from dataclasses import dataclass

@dataclass
class LLMAPIConfig:
    """大语言模型 (LLM) API 与裁判 Prompt 统一配置类"""

    # =========================================================
    # 1. DeepSeek API
    # =========================================================
    base_url: str = os.environ.get(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    )

    deepseek_api_key: str = os.environ.get(
        "DEEPSEEK_API_KEY",
        "sk-ac9b71eb104d4dfc8adbe52ce6a3d349",
    )

    judge_model_name: str = os.environ.get(
        "DEEPSEEK_MODEL",
        "deepseek-v4-flash",
    )

    # =========================================================
    # 2. 请求参数
    # =========================================================
    temperature: float = float(
        os.environ.get(
            "DEEPSEEK_TEMPERATURE",
            "0.0",
        )
    )

    max_tokens: int = int(
        os.environ.get(
            "DEEPSEEK_MAX_TOKENS",
            "256",
        )
    )

    timeout_seconds: float = float(
        os.environ.get(
            "DEEPSEEK_TIMEOUT_SECONDS",
            "60",
        )
    )

    retries: int = int(
        os.environ.get(
            "DEEPSEEK_RETRIES",
            "5",
        )
    )

    retry_seconds: float = float(
        os.environ.get(
            "DEEPSEEK_RETRY_SECONDS",
            "2",
        )
    )

    # --- 2. VQA-RAD 专属 LLM 裁判 Prompt ---
    #以 Norm 为主，Raw 为辅，且强调了医学上的致命错误不能宽容。
    #老提示词
    # medical_vqa_llm_judge_system_prompt: str =(
    #     "You are an expert medical AI evaluator. Your task is to evaluate the semantic equivalence "
    #     "between a 'Prediction' and a 'Ground Truth' for a given medical image 'Question'.\n\n"
    #     "Instructions:\n"
    #     "- Please judge based primarily on the 'Normalized Prediction' and 'Normalized Ground Truth'.\n"
    #     "- Use the 'Raw Prediction' only as supporting context to understand whether the model's answer differs merely due to wrapper phrases or formatting.\n\n"
    #     "Evaluation Criteria:\n"
    #     "1. 'correct': The Prediction has the exact same medical meaning as the Ground Truth. "
    #     "Accept valid medical abbreviations (e.g., 'us' for 'ultrasound'), synonyms, and different word orders. "
    #     "Ignore differences in punctuation, articles, and casing.\n"
    #     "2. 'partially_correct': The Prediction captures the main idea but misses critical specific details, "
    #     "or includes extra incorrect information that doesn't completely invalidate the main finding.\n"
    #     "3. 'incorrect': The Prediction is medically contradictory, misses the core finding, or is completely unrelated. "
    #     "CRITICAL: Laterality (left/right), anatomy, modality, pathology, or numeric mismatches MUST be judged as 'incorrect'.\n\n"
    #     "Output Format:\n"
    #     "You MUST output ONLY a valid JSON object with exactly two keys: 'reasoning' (a brief explanation of your logic) "
    #     "and 'score' (the exact string: 'correct', 'partially_correct', or 'incorrect'). "
    #     "Do NOT wrap the JSON in markdown blocks (like ```json)."
    # )
    medical_vqa_llm_judge_system_prompt: str = """You are an expert evaluator for medical visual question answering. Evaluate whether the Prediction correctly answers the given Question relative to the provided Ground Truth.

    Judge only the textual answer equivalence. Do not attempt to reinterpret the medical image, replace the Ground Truth, or introduce findings not supported by the provided information.

    Input Usage:

    * Compare both the Normalized and Raw answers.
    * Use the normalized answers to ignore harmless formatting and lexical differences.
    * The raw answers remain authoritative when normalization may have removed or altered medically meaningful information such as negation, laterality, anatomy, numbers, units, or multiple findings.
    * Base the required level of specificity on what the Question asks, rather than requiring every detail appearing in the Ground Truth to be repeated unconditionally.

    Evaluation Criteria:

    1. "correct"
       The Prediction fully and correctly answers the Question and is clinically equivalent to the Ground Truth.

    Accept:

    * Valid medical abbreviations and synonyms.
    * Equivalent anatomical or modality terminology.
    * Different word order, punctuation, articles, casing, or harmless wrapper phrases.
    * Equivalent numerical values expressed using different units.
    * Compatible additional specificity that does not contradict the Ground Truth.
    * Omission of details that are not required to answer the Question and do not change the medical conclusion.

    2. "partially_correct"
       The Prediction contains the correct core answer but is incomplete, overly broad, or misses a required non-core attribute.

    Use this label when:

    * The correct pathology or main finding is identified, but a required location, laterality, severity, count, or other secondary attribute is omitted.
    * The answer is a medically compatible but overly general version of the Ground Truth.
    * The answer contains minor extra incorrect information that does not contradict or invalidate the core answer.

    Do not use "partially_correct" merely because the wording differs from the Ground Truth.

    3. "incorrect"
       The Prediction fails to provide the correct core answer, directly contradicts the Ground Truth, answers a different question, or contains an error that changes the medical conclusion.

    Judge as "incorrect" when:

    * A stated laterality is opposite to the Ground Truth.
    * The stated anatomy, modality, pathology, or number is genuinely incompatible with the Ground Truth and is required by the Question.
    * Negation or presence/absence is reversed.
    * The answer gives a different core finding.
    * Incorrect additional information contradicts or invalidates the otherwise correct finding.

    Important distinctions:

    * Missing laterality may be "partially_correct" when the main finding is correct; explicitly stating the wrong laterality is "incorrect".
    * A broader anatomical description may be "partially_correct"; a contradictory anatomical location is "incorrect".
    * Numerically equivalent unit conversions are "correct". Reasonable rounding may be accepted when exact precision is not required.
    * A more specific answer is "correct" only when the added specificity is compatible with the Ground Truth.
    * Evaluate the answer at the specificity required by the Question.

    Output Format:
    Return ONLY one valid JSON object with exactly these two keys:
    {"reasoning":"brief explanation","score":"correct|partially_correct|incorrect"}

    Do not use markdown fences or include any text outside the JSON object."""
    
    def validate(self) -> None:
        if not self.deepseek_api_key:
            raise RuntimeError(
                "未设置 DEEPSEEK_API_KEY，"
                "请先配置环境变量。"
            )