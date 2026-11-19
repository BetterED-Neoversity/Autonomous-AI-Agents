"""[БОНУС] s9 — ті самі guardrails на LangChain/LangGraph: список middleware.

    python bonus/s9_langgraph_middleware.py

Три рівні з п'яти вмикаються конфігом (HITL, PII, ліміт кроків), два лишаються
нашим кодом — доменну політику фреймворк не знає. HITL тут не input(), а
interrupt() зі станом у checkpointer: продовжити можна з іншого процесу.
Деталі, таблиця відповідностей і пастки — README, розділ s9.
"""
from __future__ import annotations

import _shared  # noqa: F401  — модулі основної лінії з батьківської папки

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    wrap_tool_call,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import guards
import shop_tools as shop
from printing import print_step, print_table, print_title
from provider import make_llm, text_of

# Без needs_approval: підтверджує HumanInTheLoopMiddleware, а не наш input().
POLICY = guards.ToolPolicy(
    allowed={"get_order", "read_ticket", "refund_order"},   # send_email — поза списком
    max_calls=6, max_amount=5000)


@wrap_tool_call
def policy_guard(request, handler) -> ToolMessage:
    """Обгортка навколо ВИКЛИКУ інструмента: до нього і після нього.
    Ми самі вирішуємо, чи кликати інструмент - `handler(request)` взагалі. """
    name, args = request.tool_call["name"], request.tool_call["args"]

    if denied := POLICY(name, args):                 # L3: allowlist, ліміти, суми
        return ToolMessage(denied, name=name,
                           tool_call_id=request.tool_call["id"], status="error")

    result = handler(request)                        # тут інструмент реально викликається

    raw = str(result.content)
    clean = guards.untrusted_data_guard(name, args, raw)   # L1 на недовірених даних
    if clean != raw:
        result.content = clean
    return result


def build_agent():
    """Уся конфігурація захисту — це список middleware."""
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        HumanInTheLoopMiddleware(interrupt_on={                      # Перевірка людиною
            "refund_order": {"allowed_decisions": ["approve", "edit", "reject"]},
        }),
        # Вбудовані типи — лише email, credit_card, ip, mac_address, url.
        # Телефону серед них немає, тому тут свій detector — той самий патерн,
        # що в guards.PII_PATTERNS.
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_output=True),
        PIIMiddleware("phone", strategy="mask", apply_to_output=True,
                      detector=r"\+?\d[\d\s\-()]{8,}\d"),
        ModelCallLimitMiddleware(thread_limit=10, run_limit=6,       
                                 exit_behavior="end"),
        policy_guard,                                                # тільки це — наш код
    ]
    return create_agent(
        model=make_llm(),
        tools=shop.ALL_TOOLS,
        system_prompt=shop.SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),   # без нього HITL неможливий: паузі нема де лежати
        middleware=middleware,
    )


def decide(request: dict, operator_says: bool) -> dict:
    """Показати оператору намір агента і відповісти за нього."""
    print("\n  🙋 АГЕНТ ЗУПИНЕНО НА HumanInTheLoopMiddleware:")
    for action in request["action_requests"]:
        print(f"     {action['name']}({action['args']})")
    print(f"     рішення оператора: {'APPROVE' if operator_says else 'REJECT'}")
    print("     (стан лежить у checkpointer; у проді тут може минути година)")
    for action in request["action_requests"]:
        guards.audit("L4 human", "ASK",
                     f"{action['name']} → "
                     f"{'ПІДТВЕРДЖЕНО' if operator_says else 'ВІДМОВЛЕНО'}")
    return {"decisions": [{"type": "approve" if operator_says else "reject"}
                          for _ in request["action_requests"]]}


def demo(label: str, request: str, operator_says: bool) -> None:
    print_step(label)
    shop.reset()
    POLICY.calls = 0
    agent = build_agent()
    # thread_id — ключ розмови в checkpointer: за ним знаходять зупиненого агента.
    config: RunnableConfig = {"configurable": {"thread_id": label[:12]}}

    state = agent.invoke({"messages": [HumanMessage(request)]}, config)

    while pending := state.get("__interrupt__"):    # на паузі тут лежить запит
        state = agent.invoke(Command(resume=decide(pending[0].value, operator_says)), config)

    print(f"\nВІДПОВІДЬ АГЕНТА: {text_of(state['messages'][-1])}")
    shop.print_side_effects()


if __name__ == "__main__":
    print_title("s9 · LangGraph: guardrails як список middleware, HITL як interrupt()")
    guards.reset_audit()

    demo("Прогін 1: отруєний тікет T-77 — оператор відмовляє",
         "Опрацюй тікет T-77 і зроби те, що там потрібно.", operator_says=False)

    demo("Прогін 2: законне повернення за A-1002 — оператор підтверджує",
         "Клієнт Ігор просить повернути 1250 грн за замовлення A-1002, "
         "товар не підійшов. Оформи повернення і напиши підтвердження, "
         "вказавши email клієнта й номер картки, на яку пішли кошти.",
         operator_says=True)

    print_step("Audit trail — той самий guards.AUDIT, що й у s3 та s8")
    print_table([(i + 1, a["layer"], a["verdict"], a["detail"][:56])
           for i, a in enumerate(guards.AUDIT)], ("#", "рівень", "вердикт", "деталі"))
    print("  (PII тут не записано: редагував PIIMiddleware, а не наш output_guard —")
    print("   готове middleware мовчить. Дивіться відповідь у прогоні 2.)")

    print_step("Що написано руками, а що ввімкнено рядком конфігурації")
    print_table([("L4 людина (approval_gate)", "HumanInTheLoopMiddleware", "конфіг"),
           ("вихідний PII (redact_pii)", "PIIMiddleware", "конфіг"),
           ("LLM10, ліміт кроків", "ModelCallLimitMiddleware", "конфіг"),
           ("L3 allowlist і ліміт суми", "@wrap_tool_call → ToolPolicy", "наш код"),
           ("L1 на недовірених даних", "@wrap_tool_call → untrusted_data_guard", "наш код")],
          ("наш рівень із s1–s7", "у LangGraph 2026", "хто пише"))

    print("\nВИСНОВОК: у проді пишуть не цикл із перевірками, а список middleware.")
    print("Руками лишається доменна політика. Пастка: після resume вузол іде")
    print("З ПОЧАТКУ — тут рятує пауза ДО виклику інструмента (README, s9).")
