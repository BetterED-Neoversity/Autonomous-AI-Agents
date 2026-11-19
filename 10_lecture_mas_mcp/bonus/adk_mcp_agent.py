"""БОНУС. Той самий MCP-сервер у Google ADK через McpToolset.

Третій фреймворк потрібен рівно для одного висновку: бізнес-логіка не
переписується. Змінюється лише спосіб оголосити агента, а tools приїжджають
з того самого mcp_server.py.

У ADK інтеграція MCP — це McpToolset у списку tools; він сам піднімає сервер,
робить discovery і проксіює виклики. tool_filter — це той самий allowlist,
тільки заданий засобами фреймворку.

Шляхи (mcp_server.py, .env) рахуються від __file__, тому файл однаково
працює з кореня демо і з самої папки bonus. Модель — не нативний
Gemini-клієнт ADK, а LiteLlm: шлюз OpenAI-сумісний, тож підходить той самий
прийом, що й у mas_crewai.py (модель "openai/<LLM_MODEL>" + base_url/api_key
з .env у корені демо).

У класі НЕ запускається: ADK з залежностями важкий, тому в requirements.txt
його рядки закоментовані. Конфлікту з демо немає — версії перевірені в одному
оточенні. Самостійно, headless:

    # у requirements.txt розкоментувати блок «БОНУС»
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python bonus/adk_mcp_agent.py  # або з самої папки bonus/

root_agent лишено на рівні модуля, тож його може підхопити й `adk web`
(окремий необов'язковий шлях, на занятті не використовується).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters

# Корінь демо — батьківська папка bonus/. Усі шляхи рахуємо від __file__,
# а не від cwd: запуск і з кореня демо, і з bonus/ має працювати однаково.
DEMO_ROOT = Path(__file__).resolve().parent.parent

# Той самий .env, що й в основного демо. Логіка вибору провайдера — як у
# provider.py, але без його імпорту: bonus має бути самодостатнім.
load_dotenv(DEMO_ROOT / ".env")


def make_model() -> LiteLlm:
    """Модель для агента: провайдер визначає .env, а не код.

    LiteLlm розпізнає провайдера за префіксом моделі: "openai/<модель>" —
    будь-який OpenAI-сумісний шлюз (base_url + api_key з .env),
    "gemini/<модель>" — Google AI Studio.
    """
    provider = os.getenv("LLM_PROVIDER") or ("google" if os.getenv("GOOGLE_API_KEY") else "openai")
    if provider == "google":
        return LiteLlm(
            model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}",
            api_key=os.environ["GOOGLE_API_KEY"],
        )
    return LiteLlm(
        model=f"openai/{os.environ['LLM_MODEL']}",
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )


# Спеціаліст з технічних питань: тільки читання, жодних грошей.
root_agent = LlmAgent(
    model=make_model(),
    name="tech_support_agent",
    instruction=("Ти технічний агент підтримки. Статус замовлення і відповіді "
                 "з FAQ бери інструментами. Відповідай українською, покроково."),
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=[str(DEMO_ROOT / "mcp_server.py")],
                )
            ),
            # allowlist рівня фреймворку: process_refund цьому агенту не видно
            tool_filter=["check_order_status", "search_faq"],
        )
    ],
)


async def ask(query: str) -> str:
    """Headless-прогін без UI: Runner + InMemorySessionService.

    ADK асинхронний, тому ask() — корутина, а main() загортає її в
    asyncio.run. Runner як async-контекст-менеджер після прогону сам
    закриває McpToolset — інакше підпроцес mcp_server.py залишиться жити.
    """
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="support", user_id="student"
    )
    runner = Runner(
        app_name="support",
        agent=root_agent,
        session_service=session_service,
    )

    answer = ""
    async with runner:
        async for event in runner.run_async(
            user_id="student",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=query)]),
        ):
            # Прозорість як у step2: видно, що агент реально кликав MCP-tool,
            # а не вигадав відповідь.
            for call in event.get_function_calls():
                print(f"  → tool {call.name}({call.args})")
            for response in event.get_function_responses():
                print(f"  ← {response.response}")
            if event.is_final_response() and event.content and event.content.parts:
                # Частини з thought=True — це reasoning моделі (шлюз віддає
                # його окремим полем, ADK пакує в thought-частини): у відповідь
                # клієнту він не потрапляє.
                answer = "".join(
                    part.text or "" for part in event.content.parts if not part.thought
                )
    return answer


def main() -> None:
    query = "Де моє замовлення ORD-002?"
    print(f"питання: {query}")
    answer = asyncio.run(ask(query))
    print(f"\nвідповідь: {answer}")


if __name__ == "__main__":
    main()
