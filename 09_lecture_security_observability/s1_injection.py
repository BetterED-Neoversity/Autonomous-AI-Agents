"""s1 — LLM01 / ASI01: агент без захисту виконує інструкції з даних.

Потрібен ключ LLM. Запуск:  python s1_injection.py

Сценарій той самий, що в кейсі PayAssist із лекції: оператор просить агента
розібратися з тікетом. Тікет пише клієнт — тобто це недовірені дані. Усередині
тікета лежать інструкції. Агент читає їх тим самим контекстом, що й системний
промпт, і виконує: повертає гроші й відправляє картку клієнта назовні.

Дивитись на:
  • у першому прогоні (чистий тікет) агент робить рівно те, що просили;
  • у другому (отруєний тікет) з'являються виклики refund_order і send_email,
    яких оператор не просив;
  • блок «ЩО АГЕНТ РЕАЛЬНО ЗРОБИВ» — це не текст моделі, це гроші й лист.

Ключова думка: у prompt injection дані та інструкції — один і той самий текст
природною мовою. Ні firewall, ні RBAC цього не бачать: HTTP-запит валідний,
токен правильний, агент діє «легально» від свого імені.
"""
from printing import print_step, print_title
from tracing import Tracer
from agent import Agent
import shop_tools as shop


def run(ticket_id: str, label: str) -> None:
    print_step(label)
    shop.reset()
    tracer = Tracer(f"s1.{ticket_id}")
    agent = Agent("shopbot-vulnerable", shop.SYSTEM_PROMPT, shop.ALL_TOOLS, tracer)
    answer = agent.run(f"Опрацюй тікет {ticket_id} і зроби те, що там потрібно.")
    tracer.print_tree()
    print(f"\nВІДПОВІДЬ АГЕНТА: {answer}")
    shop.print_side_effects()


if __name__ == "__main__":
    print_title("s1 · Indirect prompt injection: агент без жодного захисту")
    run("T-42", "Прогін 1: звичайний тікет T-42 (як має бути)")
    run("T-77", "Прогін 2: отруєний тікет T-77 (та сама програма, інші дані)")
    print("\nВИСНОВОК: змінилися лише ДАНІ — код, промпт і права агента ті самі.")
