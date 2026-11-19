"""Крок 6. Human-in-the-loop: ризикову дію підтверджує людина.

Guardrails з кроку 5 не пропустять до грошей нікого, крім billing-агента. Але
billing цілком легально виконує повернення — і робить це сам, без жодної людини.
Для операцій із незворотними наслідками (гроші, видалення, розсилка) цього
замало: потрібен явний gate.

У LangGraph це interrupt(): виконання зупиняється прямо посеред вузла, стан
лягає у checkpointer, а керування повертається у ваш код. Далі система може
чекати рішення оператора хоч добу — граф відновиться з тієї самої точки через
Command(resume=...). Тут переривання ставить вбудований
HumanInTheLoopMiddleware, тому і картка, і формат рішення — його.

Сам gate у mas.py — це HumanInTheLoopMiddleware: він зупиняє агента ПІСЛЯ того,
як модель попросила ризиковий tool, і ДО того, як той виконається. Звідси й
формат обміну: у interrupt приходить список action_requests, назад іде список
decisions — по рішенню на кожен запитаний інструмент.

Дві обов'язкові умови, на яких спотикаються всі:
    1) compile(checkpointer=...) — без нього зберігати стан нікуди;
    2) thread_id у config — це ключ, за яким стан знайдуть при відновленні.

Скрипт проганяє один і той самий запит двічі, з різними рішеннями оператора:
approve і reject. Зверніть увагу, що thread_id у них різні — це два незалежні
діалоги.

Запуск:  python step6_hitl.py             (потрібен ключ у .env)
"""
from __future__ import annotations

import asyncio

import guardrails
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from mas import build_mas, initial_state
from printing import print_step, print_title

REQUEST = ("Замовлення ORD-001 прийшло з дефектом, хочу повернення коштів. "
           "Оформіть, будь ласка.")


async def run_with_decision(app, thread_id: str, approved: bool) -> None:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Перший виклик доходить до interrupt() і повертає керування нам.
    state = await app.ainvoke(initial_state(REQUEST), config=config)

    pending = state.get("__interrupt__")
    if not pending:
        print("Граф завершився без запиту до людини — ризикових дій не було.")
        print(f"відповідь: {state.get('answer', '')[:300]}")
        return

    # У картці оператора рівно те, що потрібно для рішення: хто просить,
    # що саме робить і з якими аргументами.
    action = pending[0].value["action_requests"][0]
    print("\nГраф ЗУПИНЕНО. Оператору показують:")
    print(f"  запит:      {action['description']}")
    print(f"  дія:        {action['name']}")
    print(f"  аргументи:  {action['args']}")
    print(f"\nрішення оператора: {'ПІДТВЕРДИТИ' if approved else 'ВІДХИЛИТИ'}")

    # Відновлення: по рішенню на кожен action_request. approve виконує виклик
    # як є, reject повертає моделі ToolMessage «людина відмовила».
    decision = {"type": "approve"} if approved else {"type": "reject"}
    final = await app.ainvoke(Command(resume={"decisions": [decision]}), config=config)

    print("\nщо робила система:")
    for event in final["events"]:
        print(f"  • {event}")
    print(f"\nвідповідь: {final['answer'][:400]}")


async def main() -> None:
    print_title("Крок 6. HITL: гроші не йдуть без людини")
    guardrails.reset_rate_limit()

    app, _ = await build_mas(secured=True, hitl=True)

    print_step("прогін 1: оператор підтверджує")
    print(f"клієнт: {REQUEST}")
    await run_with_decision(app, "refund-approve", approved=True)

    print_step("прогін 2: той самий запит, оператор відхиляє")
    print(f"клієнт: {REQUEST}")
    await run_with_decision(app, "refund-reject", approved=False)

    print_step("підсумок")
    print("Ризикових tools у системі має бути мало (guardrails.RISKY_TOOLS),")
    print("інакше HITL перетворюється на клікання «ОК» і перестає працювати.")
    print("Оператор бачить рівно те, що піде в tool: ім'я дії та аргументи.")
    print("У проді сюди ж додають ідентифікатор оператора і час — це audit trail.")


if __name__ == "__main__":
    asyncio.run(main())
