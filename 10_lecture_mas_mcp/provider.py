"""Єдине місце, де демо знає про конкретного LLM-провайдера.

Тут і створення моделі (make_llm), і читання її відповіді (text_of) — два боки
однієї абстракції. Додасте нового провайдера — правити доведеться лише цей файл.

Той самий файл, що у демо лекцій 5, 8 і 9, — щоб між лекціями нічого не
переучувати. Тут його імпортують mas.py і всі кроки, яким потрібна модель:
step2, step3, step5, step6, step7.

.env поруч зі скриптами:

    GOOGLE_API_KEY=...                  # https://aistudio.google.com/apikey
    GEMINI_MODEL=gemini-3.5-flash-lite  # необовʼязково

Будь-який OpenAI-сумісний шлюз:

    LLM_PROVIDER=openai
    OPENAI_BASE_URL=... / OPENAI_API_KEY=... / LLM_MODEL=...
"""
import logging
import os

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv(".env")

# google-genai пише попередження про automatic function calling на кожен виклик
# з tools. Для демо це шум: tool-виклики ми виконуємо самі, вручну.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

PROVIDER = os.getenv("LLM_PROVIDER") or ("google" if os.getenv("GOOGLE_API_KEY") else "openai")


def make_llm():
    """Чат-модель із підтримкою tool calling."""
    if PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # max_retries: у безкоштовного тарифу жорсткий ліміт запитів за хвилину.
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                                      max_retries=6)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.environ["LLM_MODEL"],
                      base_url=os.environ["OPENAI_BASE_URL"],
                      api_key=SecretStr(os.environ["OPENAI_API_KEY"]))


def text_of(message) -> str:
    """Текст відповіді моделі.

    Gemini через langchain повертає content списком блоків (текст + службовий
    підпис), OpenAI — рядком. Демо цікавить тільки текст.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict)).strip()
