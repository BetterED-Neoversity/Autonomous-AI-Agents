"""Крок 1. MCP-сервер сам по собі: без агента, без LLM, без ключа.

Тут ми граємо роль host-застосунку: піднімаємо mcp_server.py підпроцесом
через stdio, робимо initialize (capability negotiation) і дивимось, що сервер
про себе розповідає — tools, resources, prompts. Саме цей список побачить
модель, і саме з docstring'ів формуються схеми аргументів.

Головне, на що дивитись у виводі:
    1) три примітиви MCP присутні окремо: дії, дані, шаблони;
    2) схема аргументів згенерована з анотацій типів автоматично;
    3) check_order_status не повертає customer_email — PII відрізає сервер,
       а не агент. Агент не може злити те, чого не отримував;
    4) process_refund для ORD-003 (processing) отримує відмову від бізнес-логіки,
       хоч би як його про це просили.

Запуск:  python step1_mcp_server.py        (LLM не потрібен)
"""
from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from printing import print_step, print_table, print_title

# sys.executable, а не "python": підпроцес має стартувати тим самим інтерпретатором
# з .venv, інакше він не знайде пакет mcp. Це помилка №1 при першому запуску.
SERVER = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])


def _text(result) -> str:
    """Витягує текст із CallToolResult: вміст приходить списком блоків."""
    return "\n".join(block.text for block in result.content if block.type == "text")


async def main() -> None:
    print_title("Крок 1. MCP-сервер: tools, resources, prompts через stdio")

    # stdio_client запускає `python mcp_server.py` і дає нам два потоки.
    # ClientSession — це вже рівень протоколу поверх них.
    async with stdio_client(SERVER) as (read, write), ClientSession(read, write) as session:
        info = await session.initialize()
        print_step("initialize: хто на тому кінці")
        print(f"сервер: {info.serverInfo.name} {info.serverInfo.version}")
        print(f"протокол: {info.protocolVersion}")

        print_step("tools — дії, які агент може виконати")
        tools = (await session.list_tools()).tools
        print_table(
            [(t.name, ", ".join(t.inputSchema.get("properties", {})),
              (t.description or "").split("\n")[0]) for t in tools],
            ("tool", "аргументи", "опис (перший рядок docstring)"),
        )

        print_step("resources і prompts — контекст і шаблони, а не дії")
        resources = (await session.list_resources()).resources
        prompts = (await session.list_prompts()).prompts
        print_table([(str(r.uri), r.description or "") for r in resources], ("resource", "опис"))
        print_table([(p.name, ", ".join(a.name for a in (p.arguments or [])))
                     for p in prompts], ("prompt", "аргументи"))

        print_step("виклик tool: статус замовлення")
        order = json.loads(_text(await session.call_tool("check_order_status",
                                                         {"order_id": "ORD-001"})))
        print(json.dumps(order, ensure_ascii=False, indent=2))
        assert "customer_email" not in order, "PII не має виходити за межі сервера"
        print("\ncustomer_email у відповіді немає — сервер вирізав його ще до агента.")

        print_step("виклик ризикового tool: повернення коштів")
        ok = json.loads(_text(await session.call_tool(
            "process_refund", {"order_id": "ORD-001", "reason": "Дефект товару"})))
        print("ORD-001 (delivered):  ", json.dumps(ok, ensure_ascii=False))
        bad = json.loads(_text(await session.call_tool(
            "process_refund", {"order_id": "ORD-003", "reason": "Передумав"})))
        print("ORD-003 (processing): ", json.dumps(bad, ensure_ascii=False))
        print("\nПравило «повертаємо лише за delivered» живе у коді сервера, "
              "а не в промпті. Умовити його неможливо — нічого умовляти.")

        print_step("читання resource")
        content = (await session.read_resource("support://policy/refund")).contents[0]
        print(content.text)

    print_step("що далі")
    print("Сервер працює і протестований. Крок 2 підключає ці самі tools до агента.")


if __name__ == "__main__":
    asyncio.run(main())
