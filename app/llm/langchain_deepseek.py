from functools import lru_cache

from pydantic import SecretStr

from app.config import get_settings


class LLMConfigurationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_chat_model():
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise LLMConfigurationError("DEEPSEEK_API_KEY is required for ordinary triage.")

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover
        raise LLMConfigurationError("langchain-openai is required for DeepSeek chat model.") from exc

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=SecretStr(settings.deepseek_api_key),
        base_url=settings.deepseek_base_url,
        temperature=0.2,
    )
