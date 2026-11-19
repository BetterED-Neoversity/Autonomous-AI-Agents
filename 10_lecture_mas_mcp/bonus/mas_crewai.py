"""БОНУС. Той самий кейс у CrewAI: role-based замість графа.

Навіщо: ДЗ-3 вимагає другу реалізацію в іншому фреймворку і порівняльну
таблицю. Порівнювати треба не «що модніше», а що саме ви отримали й що
віддали:

    LangGraph  явні вузли й ребра, свій стан, свої точки перехоплення.
               Більше коду, повний контроль, guardrails вставляються будь-де.
    CrewAI     ролі, задача, ієрархічний процес. Коду менше, менеджер сам
               вирішує, кому делегувати, — але й перехопити виклик tool
               складніше: точки розширення визначає фреймворк.

Той самий MCP-сервер обслуговує обидві реалізації без жодної правки — у цьому
й сенс протоколу.

Цей файл у класі НЕ запускається: CrewAI тягне близько гігабайта залежностей,
тому в requirements.txt його рядки закоментовані. Конфлікту з демо немає —
версії перевірені в одному оточенні. Щоб спробувати самостійно:

    # у requirements.txt розкоментувати блок «БОНУС»
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python bonus/mas_crewai.py     # або з самої папки bonus/

Увага на три місця, де CrewAI найчастіше ламається:
    * Process.hierarchical вимагає manager_agent (або manager_llm);
    * менеджер не має права мати tools — CrewAI кидає виняток. Делегування
      і так його єдиний інструмент, бізнес-інструменти він отримує
      тільки через спеціалістів;
    * allow_delegation=True лише у менеджера. Якщо його поставити
      спеціалістам, вони починають делегувати одне одному по колу.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv
from mcp import StdioServerParameters

# Корінь демо — батьківська папка bonus/. Усі шляхи рахуємо від __file__,
# а не від cwd: запуск і з кореня демо, і з bonus/ має працювати однаково.
DEMO_ROOT = Path(__file__).resolve().parent.parent

# Той самий .env, що й в основного демо. Логіка вибору провайдера — як у
# provider.py, але без його імпорту: bonus має бути самодостатнім.
load_dotenv(DEMO_ROOT / ".env")


def make_llm() -> LLM:
    """Модель для агента: провайдер визначає .env, а не код.

    CrewAI розпізнає провайдера за префіксом моделі: "openai/<модель>" —
    будь-який OpenAI-сумісний шлюз (base_url + api_key з .env),
    "gemini/<модель>" — Google AI Studio (для нього потрібне crewai[google-genai]).
    """
    provider = os.getenv("LLM_PROVIDER") or ("google" if os.getenv("GOOGLE_API_KEY") else "openai")
    if provider == "google":
        return LLM(model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")
    return LLM(
        model=f"openai/{os.environ['LLM_MODEL']}",
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )


def main() -> None:
    # MCPServerAdapter піднімає той самий mcp_server.py і віддає його tools
    # у вигляді, зрозумілому CrewAI.
    # kickoff має відбутися всередині with: після виходу сервер уже закритий.
    server = StdioServerParameters(command=sys.executable,
                                   args=[str(DEMO_ROOT / "mcp_server.py")])

    with MCPServerAdapter(server) as mcp_tools:
        by_name = {tool.name: tool for tool in mcp_tools}

        # Той самий principle of least privilege, що й у guardrails.TOOL_PERMISSIONS:
        # роль отримує рівно свій набір інструментів, process_refund — лише у
        # billing. Координатор не отримує жодного: у Process.hierarchical
        # менеджеру tools заборонені, його єдиний «інструмент» — делегування.
        triage = Agent(
            role="Координатор підтримки",
            goal="Визначити тип звернення і делегувати його правильному спеціалісту",
            backstory="Ти приймаєш звернення першим. Грошових операцій не виконуєш.",
            llm=make_llm(),
            allow_delegation=True,
            verbose=True,
        )

        billing = Agent(
            role="Спеціаліст з оплат",
            goal="Розібратися з оплатами, рахунками і поверненнями коштів",
            backstory="Ти працюєш із фінансовими запитами і перевіряєш факти інструментами.",
            llm=make_llm(),
            tools=[by_name["check_order_status"], by_name["check_payment"],
                   by_name["process_refund"], by_name["search_faq"]],
            allow_delegation=False,
            verbose=True,
        )

        tech = Agent(
            role="Технічний спеціаліст",
            goal="Допомогти з технічною проблемою або статусом доставки",
            backstory="Ти пояснюєш рішення коротко й покроково. Грошей не торкаєшся.",
            llm=make_llm(),
            tools=[by_name["check_order_status"], by_name["search_faq"]],
            allow_delegation=False,
            verbose=True,
        )

        # У hierarchical-процесі task не прив'язують до агента: кому його
        # виконувати, вирішує менеджер. У LangGraph цю маршрутизацію ви
        # писали руками — тут її віддали фреймворку.
        support_task = Task(
            description=("Оброби звернення клієнта: {query}\n"
                         "1. Визнач тип: billing чи technical.\n"
                         "2. Делегуй роботу спеціалісту.\n"
                         "3. Поверни фінальну відповідь українською."),
            expected_output="Коротка відповідь українською з рішенням або наступним кроком.",
        )

        crew = Crew(
            agents=[billing, tech],
            tasks=[support_task],
            process=Process.hierarchical,
            manager_agent=triage,
            verbose=True,
        )

        result = crew.kickoff(inputs={"query": "Подвійне списання за замовленням ORD-001"})
        # Guardrails з guardrails.py тут теж застосовні: input_guardrail — перед
        # kickoff, output_guardrail — до result. Рівень tools доведеться вішати
        # на обгортки інструментів, бо своєї точки перехоплення виклику
        # CrewAI не дає так само явно, як LangGraph.
        print(result)


if __name__ == "__main__":
    main()
