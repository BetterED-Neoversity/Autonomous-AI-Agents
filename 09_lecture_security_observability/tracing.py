"""Мінімальний трасувальник: «Langfuse на 100 рядків».

Модель даних та сама, що і в OpenTelemetry — дерево spanʼів, у кожного вхід,
вихід, тривалість і атрибути. Справжній Langfuse/LangSmith відрізняється тим,
що дерево їде по мережі в UI, а не в консоль; s4_observability.py показує
обидва боки, а s4b/s4c підмінюють цей клас платформенними обгортками.

Читати згори вниз: спершу дані (Span), далі збір (Tracer.span, event, walk),
потім метрики і лише наприкінці — друк. Ядро — це перші два блоки, решта
похідна від них.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

# Ціна за 1M токенів у доларах — цифри умовні, потрібні лише щоб метрика «cost»
# у трасі була не нулем. Замініть на прайс свого провайдера.
PRICE_IN, PRICE_OUT = 0.10, 0.40


def cost_parts(attrs: dict) -> tuple[float, float]:
    """Ціна входу і виходу окремо — саме так її чекають Langfuse/LangSmith."""
    return (attrs.get("in_tok", 0) / 1e6 * PRICE_IN,
            attrs.get("out_tok", 0) / 1e6 * PRICE_OUT)


def cost_of(attrs: dict) -> float:
    """Ціна одного виклику. Формула живе тут одна на весь проєкт."""
    return round(sum(cost_parts(attrs)), 6)

# Спрацювання guardrail не заводить окремий span: хук не бачить трасувальника
# (сигнатури в agent.py — це чисті функції), тому лишає прапорець на спані, у
# якому спрацював. Список нижче — повний перелік таких прапорців, по ньому
# рахується метрика guard_triggers. Додали хук з новим прапорцем — допишіть сюди.
# Людські рішення і рішення рівнів L1-L4 живуть окремо, у guards.AUDIT.
GUARD_FLAGS = ("blocked",        # before_tool  заборонив виклик
               "sanitized",      # after_tool   почистив недовірені дані
               "short_circuit",  # before_model відповів замість моделі
               "rewritten")      # after_model  переписав сиру відповідь

ICONS = {"agent": "🤖", "llm": "🧠", "tool": "🔧", "eval": "📏"}

TREE_LIMIT = 110   # скільки символів тексту лишаємо у консольному дереві
OTLP_LIMIT = 80    # у payload коротше: там дивляться на структуру, не на текст


@dataclass
class Span:
    name: str
    kind: str                      # agent | llm | tool | eval — і все, інших немає
    inp: str = ""
    out: str = ""
    attrs: dict = field(default_factory=dict)
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    children: list["Span"] = field(default_factory=list)

    @property
    def ms(self) -> float:
        """Тривалість. У незакритого spanʼа — скільки минуло на цю мить."""
        return ((self.end or time.perf_counter()) - self.start) * 1000


class Tracer:
    """Збирає дерево spanʼів, рахує метрики і вміє надрукувати те й те."""

    def __init__(self, trace_name: str):
        self.name = trace_name
        self.root = Span(trace_name, "agent")
        # Вкладеність визначає порядок входу у with, а не хто кого викликав.
        # Отже: один потік, без async. Справжні трасувальники тримають цей стек
        # у contextvars — саме тому s4b бере вкладеність з контексту OpenTelemetry.
        self._stack = [self.root]

    # ── збір ────────────────────────────────────────────────────────────────
    @contextmanager
    def span(self, name: str, kind: str, inp: str = "", **attrs):
        span = Span(name, kind, inp=inp, attrs=dict(attrs))
        self._stack[-1].children.append(span)
        self._stack.append(span)
        try:
            yield span
        finally:
            span.end = time.perf_counter()
            # Токени викликач проставляє всередині with, тому ціну рахуємо тут,
            # на виході. Далі вона їде в дерево і в OTLP сама — обидва друкують
            # усе, що лежить в attrs.
            if "in_tok" in span.attrs or "out_tok" in span.attrs:
                span.attrs["cost_usd"] = cost_of(span.attrs)
            self._stack.pop()

    def event(self, name: str, kind: str, inp: str = "", out: str = "", **attrs) -> Span:
        """Миттєва подія без вкладеності — спрацювання guardrail, рішення людини."""
        span = Span(name, kind, inp=inp, out=out, attrs=dict(attrs))
        span.end = span.start
        if "in_tok" in span.attrs or "out_tok" in span.attrs:
            span.attrs["cost_usd"] = cost_of(span.attrs)
        self._stack[-1].children.append(span)
        return span

    def walk(self, span: Span | None = None):
        """Усі spanʼи дерева, згори вниз. Основа і метрик, і друку."""
        span = span or self.root
        yield span
        for child in span.children:
            yield from self.walk(child)

    # ── метрики ─────────────────────────────────────────────────────────────
    def metrics(self) -> dict:
        spans = list(self.walk())
        tok_in = sum(span.attrs.get("in_tok", 0) for span in spans)
        tok_out = sum(span.attrs.get("out_tok", 0) for span in spans)
        return {
            "latency_ms": round(self.root.ms),
            # рахуємо за наявністю токенів: критик — теж виклик LLM, просто в ролі eval
            "llm_calls": sum("in_tok" in span.attrs for span in spans),
            "tool_calls": sum(span.kind == "tool" for span in spans),
            "guard_triggers": sum(any(span.attrs.get(f) for f in GUARD_FLAGS)
                                  for span in spans),
            "tokens_in": tok_in,
            "tokens_out": tok_out,
            "cost_usd": round(sum(span.attrs.get("cost_usd", 0) for span in spans), 6),
        }

    # ── вивід ───────────────────────────────────────────────────────────────
    def print_tree(self) -> None:
        self.root.end = self.root.end or time.perf_counter()
        print(f"\nTRACE {self.name}")
        self._print(self.root, "", True)

    def _print(self, span: Span, prefix: str, last: bool) -> None:
        is_root = span is self.root
        head = "" if is_root else prefix + ("└─ " if last else "├─ ")
        body = prefix + ("" if is_root else ("   " if last else "│  "))
        attrs = " ".join(f"{k}={v}" for k, v in span.attrs.items())
        print(f"{head}{ICONS.get(span.kind, '·')} {span.name}  [{span.ms:.0f} ms] {attrs}")
        for text, tag in ((span.inp, "in "), (span.out, "out")):
            if text:
                print(f"{body}     {tag}: {_short(text)}")
        for i, child in enumerate(span.children):
            self._print(child, body, i == len(span.children) - 1)

    def print_metrics(self) -> None:
        print("\nМЕТРИКИ ТРАСИ")
        for k, v in self.metrics().items():
            print(f"  {k:<16} {v}")

    def to_otlp(self) -> str:
        """Те саме дерево у вигляді, близькому до OTLP-payload (спрощено)."""
        def conv(span: Span) -> dict:
            d = {"name": span.name, "attributes": {"span.kind": span.kind,
                                                   "duration_ms": round(span.ms, 1),
                                                   **span.attrs}}
            if span.inp:
                d["attributes"]["input"] = _short(span.inp, OTLP_LIMIT)
            if span.out:
                d["attributes"]["output"] = _short(span.out, OTLP_LIMIT)
            if span.children:
                d["children"] = [conv(c) for c in span.children]
            return d
        return json.dumps(conv(self.root), ensure_ascii=False, indent=2)


def _short(text: str, limit: int = TREE_LIMIT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " …"


def tool_calls_of(tracer: Tracer) -> list[tuple[str, bool]]:
    """[(назва інструмента, чи заблокований guardrailом)] — для evals у s6."""
    return [(span.name.removeprefix("tool."), bool(span.attrs.get("blocked")))
            for span in tracer.walk() if span.kind == "tool"]
