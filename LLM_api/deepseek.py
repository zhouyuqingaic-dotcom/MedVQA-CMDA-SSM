import time

from openai import OpenAI


class DeepSeekClient:
    """DeepSeek OpenAI-compatible API 客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_seconds: float = 60.0,
        retries: int = 5,
        retry_seconds: float = 2.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "DeepSeek API Key 不能为空。"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_seconds = retry_seconds

    @classmethod
    def from_config(
        cls,
        cfg,
    ) -> "DeepSeekClient":
        cfg.validate()

        return cls(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.base_url,
            model=cfg.judge_model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_seconds=cfg.timeout_seconds,
            retries=cfg.retries,
            retry_seconds=cfg.retry_seconds,
        )

    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        for attempt in range(self.retries):
            try:
                completion = (
                    self.client
                    .chat
                    .completions
                    .create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,

                        # 强制输出合法 JSON。
                        response_format={
                            "type": "json_object"
                        },

                        # V4 默认开启 thinking，
                        # Judge 评测中显式关闭。
                        extra_body={
                            "thinking": {
                                "type": "disabled"
                            }
                        },
                    )
                )

                content = (
                    completion
                    .choices[0]
                    .message
                    .content
                )

                # DeepSeek 官方说明 JSON 模式下
                # 偶尔可能返回空内容，因此纳入重试。
                if (
                    not content
                    or not content.strip()
                ):
                    raise ValueError(
                        "DeepSeek 返回了空内容。"
                    )

                return content.strip()

            except Exception as exc:
                print(
                    "⚠️ [DeepSeek API 请求报错] "
                    f"第 {attempt + 1} 次尝试失败: "
                    f"{exc}"
                )

                if attempt < self.retries - 1:
                    time.sleep(
                        self.retry_seconds
                    )

        print(
            "❌ DeepSeek API 达到最大重试次数，"
            "放弃请求。"
        )

        return "ERROR"
