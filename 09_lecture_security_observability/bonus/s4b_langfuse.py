"""[БОНУС до s4] s4b — та сама траса, що в s4, але в справжньому Langfuse.

Потрібні ключ LLM і запущений Langfuse. Запуск:  python bonus/s4b_langfuse.py

Уся інтеграція — клас LangfuseTracer: підклас Tracer з tracing.py з одним
перевизначеним методом. Агент, guardrails і домен з s3 не змінені жодним рядком.

Дивитись на:
  • kind нашого span → тип observation: llm→generation, tool→tool, agent→agent;
  • usage_details і cost_details — токени й гроші рахує вже платформа;
  • propagate_attributes — session_id і user_id на всій трасі й на всіх дочірніх;
  • score_trace — метрики з s6 лягають прямо на трасу.

Тільки happy path: без перевірок ключів і фолбеків.

Langfuse локально (деталі — langfuse_docker/README.md):

    cd langfuse_docker && docker compose up -d

Три рядки в .env поруч зі скриптами; ключі Langfuse створює сам при старті:

    LANGFUSE_HOST=http://localhost:3000
    LANGFUSE_PUBLIC_KEY=pk-lf-shopbot-demo-0000
    LANGFUSE_SECRET_KEY=sk-lf-shopbot-demo-0000
"""
from __future__ import annotations

import _shared  # noqa: F401  — модулі основної лінії з батьківської папки

import os
from contextlib import contextmanager

from langfuse import get_client, propagate_attributes
from langfuse.api import ScoreDataType

from printing import print_step, print_title
from tracing import Span, Tracer, cost_parts
import guards
import shop_tools as shop
from s3_secured_agent import secured_agent   # він же підвантажує .env

TRACE_NAME = "s4b.ticket-77"
SESSION_ID = "ticket-77"          # усі траси одного звернення → одна сесія
USER_ID = "operator-12"           # хто працював з агентом
REQUEST = "Опрацюй тікет T-77 і зроби те, що там потрібно."

# Без назви моделі Langfuse не порахує ціну generation.
MODEL = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "gemini-3.5-flash-lite"

# kind нашого Span → тип observation Langfuse. Онтології збігаються, бо обидві
# виросли з OpenTelemetry GenAI semantic conventions.
AS_TYPE = {"agent": "agent", "llm": "generation", "tool": "tool", "eval": "evaluator"}


def cost_fields(span: Span) -> dict:
    """Поля, з яких Langfuse рахує вартість llm-виклику: модель, токени, ціна.

    Для не-llm спанів — порожньо: у tool і agent вартості немає.
    """
    if span.kind != "llm":
        return {}
    cost_in, cost_out = cost_parts(span.attrs)
    return {"model": MODEL,
            "usage_details": {"input": span.attrs.get("in_tok", 0),
                              "output": span.attrs.get("out_tok", 0)},
            "cost_details": {"input": cost_in, "output": cost_out}}


class LangfuseTracer(Tracer):
    """Tracer, який паралельно пише те саме дерево в Langfuse."""

    def __init__(self, trace_name: str, client):
        super().__init__(trace_name)
        self.lf = client

    @contextmanager
    def span(self, name: str, kind: str, inp: str = "", **attrs):
        # 1. Span у памʼяті — з нього консольне дерево і метрики.
        with super().span(name, kind, inp=inp, **attrs) as span:
            # 2. Той самий span як observation. Вкладеність SDK бере з контексту
            #    OpenTelemetry, parent передавати не треба.
            with self.lf.start_as_current_observation(
                    name=name, as_type=AS_TYPE[kind]) as observation:
                try:
                    yield span          # тут працює агент і заповнює span
                finally:
                    # 3. Вхід, вихід і атрибути викликач проставляє всередині
                    #    with — тому в Langfuse вони їдуть на виході.
                    observation.update(input=span.inp, output=span.out,
                                       metadata=span.attrs, **cost_fields(span))


print_title("s4b · Той самий агент у справжньому Langfuse")

lf = get_client()                 # хост і ключі бере з LANGFUSE_* у .env
shop.reset()
guards.reset_audit()
tracer = LangfuseTracer(TRACE_NAME, lf)

print_step("Прогін: змінився лише клас трасувальника")
# propagate_attributes викликати якомога раніше: спани, створені до входу
# в контекст, атрибутів не отримають і випадуть з агрегацій у дашборді.
with lf.start_as_current_observation(name=TRACE_NAME, as_type="agent",
                                     input=REQUEST) as root, \
        propagate_attributes(session_id=SESSION_ID, user_id=USER_ID,
                             tags=["lecture-09", "shopbot", "secured"]):
    # Виклик самого агента
    answer = secured_agent(tracer).run(REQUEST)
    
    metrics = tracer.metrics()
    root.update(output=answer, metadata=metrics)

    # Журнал guardrails — подіями на трасі: у UI видно прямо в дереві.
    for record in guards.AUDIT:
        root.create_event(name=f"guardrail.{record['layer']}", input=record["detail"],
                          output=record["verdict"],
                          level="DEFAULT" if record["verdict"] == "ALLOW" else "WARNING")

    # Оцінки на трасу — ті самі метрики, що в s6, тільки в дашборді.
    root.score_trace(name="guard_triggers", value=metrics["guard_triggers"],
                     comment="скільки разів спрацював захист")
    root.score_trace(name="incident_free", value=1.0 if not shop.SIDE_EFFECTS else 0.0,
                     data_type=ScoreDataType.BOOLEAN,
                     comment="чи не сталося побічних ефектів")

    trace_id = lf.get_current_trace_id()

lf.flush()                        # SDK шле батчами у фоні, а скрипт завершується

tracer.print_tree()
print(f"\nВІДПОВІДЬ: {answer}")
tracer.print_metrics()

print_step("Де дивитись")
print(f"  траса: {lf.get_trace_url(trace_id=trace_id)}")
print(f"  сесія {SESSION_ID} — вкладка Sessions, користувач {USER_ID} — вкладка Users")
print("\n  Чого немає в консольній трасі з s4:")
print("    • історія прогонів і пошук по ній;")
print("    • агрегати: latency P95, вартість за день, спрацювання guardrails;")
print("    • scores поруч із прогоном;")
print("    • розріз по сесіях і користувачах.")
