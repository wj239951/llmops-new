# 调用 DeepSeek 官方 API（兼容 OpenAI SDK）。
from openai import OpenAI
from app.core.config import settings

class DeepSeekService:
    def chat(self, query: str, system_prompt: str, model_name: str | None = None,
             temperature: float = 0.7, top_p: float = 0.9,
             history: list[dict] | None = None) -> dict:
        if not settings.DEEPSEEK_API_KEY or settings.DEEPSEEK_API_KEY == "your_deepseek_api_key":
            raise ValueError("DEEPSEEK_API_KEY 未配置，请在 backend/.env 中填写真实 DeepSeek Key。")
        model = model_name or settings.DEEPSEEK_MODEL
        client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)

        # 会话记忆：system -> 历史(用户/助手) -> 当前问题
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history or []:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            timeout=100,
        )
        message = response.choices[0].message
        answer = message.content or ""
        # deepseek-reasoner 会把思考过程放在 reasoning_content 字段里，deepseek-chat 没有
        reasoning = getattr(message, "reasoning_content", "") or ""
        return {
            "answer": answer,
            "reasoning": reasoning,
            "model_provider": "deepseek",
            "model_name": model,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

    def chat_stream(
        self,
        query: str,
        system_prompt: str,
        model_name: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        history: list[dict] | None = None,
    ):
        """流式生成：先产出 reasoning 分片，再产出 content 分片，最后产出 done 汇总。

        产出格式：{"reasoning": str} / {"content": str} / {"done": True, ...汇总...}
        """
        if not settings.DEEPSEEK_API_KEY or settings.DEEPSEEK_API_KEY == "your_deepseek_api_key":
            raise ValueError("DEEPSEEK_API_KEY 未配置，请在 backend/.env 中填写真实 DeepSeek Key。")
        model = model_name or settings.DEEPSEEK_MODEL
        client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history or []:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            timeout=100,
            stream=True,
            # 让服务端在最后一个分片带上 usage（deepseek 官方支持）
            stream_options={"include_usage": True},
        )

        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        total_tokens = 0
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                total_tokens = usage.total_tokens or total_tokens
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            # deepseek-reasoner 的思考过程在 reasoning_content 里；deepseek-chat 没有
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                yield {"reasoning": reasoning}
            content = getattr(delta, "content", None)
            if content:
                answer_parts.append(content)
                yield {"content": content}

        yield {
            "done": True,
            "answer": "".join(answer_parts),
            "reasoning": "".join(reasoning_parts),
            "model_provider": "deepseek",
            "model_name": model,
            "total_tokens": total_tokens,
        }
