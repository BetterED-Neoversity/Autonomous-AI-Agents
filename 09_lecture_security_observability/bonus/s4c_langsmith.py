"""[БОНУС до s4] s4c — LangSmith: трасування без жодного рядка коду.

Потрібні ключ LLM і ключ LangSmith. Запуск:  python bonus/s4c_langsmith.py

Три змінні оточення — і LangChain трасується сам. Агент, guardrails і домен
з s3 не змінені жодним рядком: увесь «код інтеграції» нижче — це два
os.environ, які в проді просто лежать у .env.

Дивитись на UI, а не на консоль — і читати його як результат автоінструментації:
ми не написали жодного рядка трасування, тому платформа показує рівно те, що
бачить сам LangChain. Run-и плоскі й усі кореневі (цикл агента живе в agent.py),
а викликів, які зупинив guardrail, немає взагалі: він спрацював до .invoke().
Це не вада LangSmith — з тим самим нулем коду Langfuse дасть те саме.

Зате в кожному llm-run-і задарма повний промпт, сира відповідь із tool_calls
і токени. «Нуль коду» дає деталі, але не дає структури; структуру додає не
платформа, а ваш код: адаптер на два десятки рядків, як у s4b.

Ключ: https://smith.langchain.com → Settings → API Keys (безкоштовний тариф —
5000 трас на місяць). У .env:  LANGSMITH_API_KEY=lsv2_pt_...
US акаунт: ще й LANGSMITH_ENDPOINT=https://api.smith.langchain.com
"""
import _shared  # noqa: F401  — модулі основної лінії з батьківської папки

import os

from printing import print_step, print_title
from tracing import Tracer, tool_calls_of
import guards
import shop_tools as shop
from s3_secured_agent import secured_agent   # він же підвантажує .env

PROJECT = "shopbot-s4c-2"

if not os.getenv("LANGSMITH_API_KEY"):
    raise SystemExit("\n⚠️  Немає LANGSMITH_API_KEY — див. докстрінг цього файла.")

# Уся інтеграція. У проді ці рядки живуть у .env, а не в коді.
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = PROJECT

print_title("s4c · LangSmith: трасування за нуль рядків коду")

shop.reset()
guards.reset_audit()
tracer = Tracer("s4c.local")        # консольне дерево — щоб було з чим порівняти
answer = secured_agent(tracer).run("Опрацюй тікет T-77 і зроби те, що там потрібно.")

tracer.print_tree()
print(f"\nВІДПОВІДЬ: {answer}")

print_step("Тепер відкрийте той самий прогін у LangSmith")
api = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
print(f"  {api.replace('//eu.api.', '//eu.').replace('//api.', '//')} → Projects → {PROJECT}")
# Список беремо з власного дерева: у LangSmith цих викликів якраз і не буде.
blocked = list(dict.fromkeys(name for name, was_blocked in tool_calls_of(tracer)
                             if was_blocked))
missing = (f"немає викликів, які зупинив guardrail ({', '.join(blocked)})" if blocked
           else "виклики, зупинені guardrailом, у трасу не потрапляють")

print("\n  Ми не налаштували нічого, крім env-змінної, тому в UI — лише те,")
print("  що бачить сам LangChain:")
print("    −  плоскі run-и замість дерева агента;")
print(f"    −  {missing};")
print("    +  повний промпт, tool_calls і токени — задарма.")
print("\n  Структуру додає не платформа, а ваш код: див. s4b.")
