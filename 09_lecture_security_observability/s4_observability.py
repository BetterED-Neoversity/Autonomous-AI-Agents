"""s4 — Observability: три стовпи (traces, metrics, logs) на живому прогоні.

Потрібен ключ LLM. Запуск:  python s4_observability.py

Трасувальник тут свій, на 60 рядків (tracing.py) — не тому, що так треба в проді,
а щоб було видно модель даних, яку LangSmith і Langfuse ховають за SDK: дерево
spanʼів, у кожного kind, вхід, вихід, тривалість, атрибути. Наприкінці те саме
дерево друкується як OTLP-подібний payload — саме такий JSON їде в Langfuse,
Jaeger чи Grafana Tempo.

Дивитись на:
  • trace відповідає на питання «ЧОМУ агент так вчинив» — видно кожен крок;
  • метрики зібрані з того самого дерева: latency, токени, вартість, guardrails;
"""
from printing import print_step, print_table, print_title
from tracing import Tracer
import guards
import shop_tools as shop
from s3_secured_agent import secured_agent

if __name__ == "__main__":
    print_title("s4 · Observability: traces → metrics → logs")

    shop.reset()
    guards.reset_audit()
    tracer = Tracer("s4.ticket-77")

    print_step("Стовп 1 · TRACE: повний шлях запиту деревом spanʼів")
    answer = secured_agent(tracer).run("Опрацюй тікет T-77 і зроби те, що там потрібно.")
    tracer.print_tree()
    print(f"\nВІДПОВІДЬ: {answer}")

    print_step("Стовп 2 · METRICS: рахуються з того самого дерева")
    tracer.print_metrics()
    print("\n  Пороги з лекції: P95 latency > 15 с · error rate > 5% ·")
    print("  injection attempts > 10/хв · cost > 2× baseline → alert.")

    print_step("Стовп 3 · LOGS: семантичні, а не «error 500»")
    print("  Журнал guardrails цього прогону — саме він робить post-mortem можливим:")
    print_table([(i + 1, a["layer"], a["verdict"], a["detail"][:56])
           for i, a in enumerate(guards.AUDIT)], ("#", "рівень", "вердикт", "деталі"))

    print_step("Той самий trace як OTLP-payload (це і летить у Langfuse/Jaeger/Tempo)")
    print(tracer.to_otlp())

