"""Крок 2. Один агент + MCP tools: найкоротший шлях від сервера до моделі.

Ще не мультиагентна система — спершу треба побачити, що tools із MCP взагалі
доїжджають до LLM. MultiServerMCPClient робить із MCP-сервера звичайні
LangChain-tools, які приймає create_agent (create_react_agent вважається
застарілим).

«Multi» у назві не випадкове: у словнику може бути кілька серверів одночасно —
свій, партнерський, публічний. Звідси й головний ризик MCP: інструменти
приходять із зовнішнього джерела разом зі своїми описами, а описи потрапляють
у контекст моделі. Недовірений сервер = ін'єкція через metadata. Крок 4
показує, що з цим робити.

Запуск:  python step2_mcp_agent.py         (потрібен ключ у .env)
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from printing import print_step, print_title
from provider import make_llm

# Абсолютний шлях і sys.executable — дві типові причини «tools не підвантажились».
MCP_SERVERS: dict[str, Any] = {
    "support": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["mcp_server.py"],
    }
}


async def main() -> None:
    print_title("Крок 2. Агент викликає MCP-tools")

    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()

    print_step("що приїхало із сервера")
    for tool in tools:
        print(f"  {tool.name:20} {tool.description.splitlines()[0]}")
    print("\nЦе вже звичайні LangChain-tools: агент не знає, що за ними MCP.")

    agent = create_agent(
        model=make_llm(),
        tools=tools,
        system_prompt=(
            "Ти агент підтримки інтернет-магазину. Відповідай українською, коротко. "
            "Факти про замовлення, оплати та правила бери ЛИШЕ з інструментів, "
            "нічого не вигадуй."
        ),
    )

    for question in ["Що зі статусом замовлення ORD-002?",
                     "Скільки днів іде доставка?"]:
        print_step(f"питання: {question}")
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        for message in result["messages"]:
            calls = getattr(message, "tool_calls", None)
            if calls:
                for call in calls:
                    print(f"  → tool {call['name']}({call['args']})")
            elif message.type == "tool":
                print(f"  ← {message.content[:120]}")
        print(f"\nвідповідь: {result['messages'][-1].content}")

    print_step("чого тут ще немає")
    print("Один агент має доступ до ВСІХ чотирьох tools, зокрема до process_refund.")
    print("Далі: крок 3 розділяє ролі, крок 4 обмежує права кожної.")


if __name__ == "__main__":
    asyncio.run(main())
