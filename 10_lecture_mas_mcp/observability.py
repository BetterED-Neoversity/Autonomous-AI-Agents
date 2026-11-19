"""Три питання до LangSmith, на які демо відповідає кодом, а не скріншотом.

    trace_report(run_id) — де подивитись дерево прогону і скільки він коштував
    print_trace_tree(run_id) — те саме дерево, але одразу в консолі
    ensure_model_price() — щоб «скільки коштував» узагалі мало значення

Усе інше LangSmith робить без жодного рядка коду: LangGraph, LangChain і
MCP-tools інструментовані самі, guardrails несуть @traceable у guardrails.py.
Тут — лише те, чого автоматика не знає: ціна вашої моделі і зворотний зв'язок
у консоль.

Про ціни. LangSmith рахує вартість сам, але тільки для моделей зі свого
прайс-листа, і зіставляє їх парою (provider, model_name). Публічну модель у
її рідного провайдера він упізнає одразу. А от коли запити йдуть через
корпоративний LLM-gateway (OPENAI_BASE_URL у цьому демо), провайдер у трейсі
записується як «openai» — пари немає, і вартість лишається порожньою. Саме це
місце й закриває ensure_model_price: ціну свого шлюзу ви реєструєте один раз,
далі кожен наступний прогін має вартість. Уже записані прогони не
перераховуються — LangSmith не робить backfill.

Модуль свідомо на requests, а не на langsmith SDK: кілька викликів до
задокументованого REST API читаються без знання клієнта, і їх видно цілком.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv(".env")

TRACING = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
API = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
UI = "https://smith.langchain.com"
KEY = os.getenv("LANGSMITH_API_KEY", "")


def _get(path: str, **params):
    r = requests.get(f"{API}/api/v1/{path}", headers={"x-api-key": KEY},
                     params=params, timeout=20)
    return r.json() if r.ok else None


def _post(path: str, **body):
    r = requests.post(f"{API}/api/v1/{path}", headers={"x-api-key": KEY},
                      json=body, timeout=20)
    return r.json() if r.ok else None


def _wait_root(run_id, tries: int = 15) -> dict | None:
    """Чекає, поки корінь прогону долетить до LangSmith і дорахується.

    Мало дочекатись end_time: токени й вартість гілок LangSmith підсумовує ще
    кілька секунд після закриття кореня, і прочитане надто рано — це сума
    випадкової частини прогону. Ознака готовності — два однакові заміри.
    """
    root, seen = None, None
    for _ in range(tries):
        run = _get(f"runs/{run_id}")
        if run and run.get("end_time"):
            root = run
            mark = (run.get("total_tokens"), run.get("total_cost"))
            if mark == seen:
                break
            seen = mark
        time.sleep(2)
    return root


def trace_report(run_id) -> dict:
    """Посилання на дерево прогону, токени і вартість — з LangSmith.

    Порожній словник означає «трасування вимкнене» і ніяк не заважає демо.
    Прогін, зупинений вхідним guardrail, моделі не звав: токенів і ціни в нього
    немає, а посилання є — у трасі видно, ЧОМУ він зупинився.
    """
    if not (TRACING and KEY):
        return {}
    root = _wait_root(run_id)
    if root is None:
        return {}
    report = {"url": UI + root["app_path"], "tokens": root.get("total_tokens")}
    if root.get("total_cost") is not None:
        report["cost"] = float(root["total_cost"])
    return report


# Поля span'а, які цікавлять консоль. Просити все — це десятки кілобайт inputs
# та outputs на кожен вузол, яких ми однаково не друкуємо.
_SPAN_FIELDS = ["name", "run_type", "dotted_order", "start_time", "end_time",
                "total_tokens", "total_cost", "tags", "error", "extra"]

# Кожен middleware — це вузол у графі агента, а отже span. Змісту в них немає,
# глибину вони дають, тому в compact-режимі ховаємо, а дітей піднімаємо вище.
_PLUMBING = {"model", "tools", "before_agent", "after_agent",
             "before_model", "after_model", "awrap_model_call", "awrap_tool_call"}


def _is_plumbing(name: str) -> bool:
    return (name in _PLUMBING or name.rsplit(".", 1)[-1] in _PLUMBING
            or name.endswith("-agent"))     # підграф create_agent під вузлом графа


def _dt(value: str) -> datetime:
    """Час зі LangSmith у datetime. 'Z' наприкінці Python до 3.11 не розумів."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _wait_spans(trace_id: str, tries: int = 8) -> list[dict]:
    """Чекає, поки дерево долетить цілком.

    Закритий корінь не означає, що всі нащадки вже в базі: хвіст приїжджає на
    секунду-дві пізніше. Ознака готовності — два однакові заміри поспіль.
    """
    spans, seen = [], 0
    for _ in range(tries):
        answer = (_post("runs/query", trace=trace_id, select=_SPAN_FIELDS,
                        limit=100) or {}).get("runs", [])
        if answer:                       # порожня відповідь — це збій або лаг,
            spans = answer               # а не «дерево з нуля вузлів»
            if len(spans) == seen:
                break
            seen = len(spans)
        time.sleep(2)
    return spans


def print_trace_tree(run_id, *, compact: bool = True) -> None:
    """Друкує дерево прогону — щоб на занятті не перемикатись у браузер.

    compact лишає ролі, guardrails, модель і tools; compact=False показує все,
    як у LangSmith, — видно, скільки з дерева службові вузли фреймворка.

    Ієрархію дає dotted_order: це шлях від кореня, батько — усе, що ліворуч
    від останньої крапки.
    """
    if not (TRACING and KEY):
        return
    root = _wait_root(run_id)
    if root is None:
        return
    spans = sorted(_wait_spans(str(root["trace_id"])), key=lambda s: s["dotted_order"])
    if not spans:
        return

    children: dict[str, list[dict]] = {}
    for span in spans[1:]:
        children.setdefault(span["dotted_order"].rsplit(".", 1)[0], []).append(span)

    inherited = set(root.get("tags") or [])
    print("\nдерево прогону:")
    print(f"  {_line(spans[0], inherited)}")
    _render(_keep(spans[0], children, compact), inherited)


def _keep(span: dict, children: dict[str, list[dict]], compact: bool) -> list:
    """Діти span'а без службових вузлів: їхніх дітей віддаємо дідові."""
    kept = []
    for child in children.get(span["dotted_order"], []):
        grandkids = _keep(child, children, compact)
        kept += grandkids if compact and _is_plumbing(child["name"]) \
            else [(child, grandkids)]
    return kept


def _render(nodes: list, inherited: set[str], prefix: str = "") -> None:
    """Малює гілки: риска йде лише там, де гілка ще триває."""
    for index, (span, kids) in enumerate(nodes):
        last = index == len(nodes) - 1
        print(f"  {prefix}{'└─ ' if last else '├─ '}{_line(span, inherited)}")
        _render(kids, inherited, prefix + ("   " if last else "│  "))


def _line(span: dict, inherited: set[str]) -> str:
    """Рядок span'а: тривалість, токени, ціна і те, що дописали ми."""
    marks = []
    if span.get("end_time"):
        ms = (_dt(span["end_time"]) - _dt(span["start_time"])).total_seconds() * 1000
        marks.append(f"{ms:.0f}ms")
    if span.get("total_tokens"):
        marks.append(f"{span['total_tokens']}tok")
    if span.get("total_cost") is not None:
        marks.append(f"${float(span['total_cost']):.6f}")   # ціна є в кожної гілки

    # Метадані від guardrails._annotate і від mas.py: заради них span і читають.
    meta = (span.get("extra") or {}).get("metadata") or {}
    marks += [f"{k}={meta[k]}" for k in ("role", "verdict", "rule") if k in meta]
    # Теги кореня успадковують усі нащадки, службові теги LangGraph — із двокрапкою.
    marks += [t for t in (span.get("tags") or []) if t not in inherited and ":" not in t]
    if span.get("error"):
        marks.append("ERROR")
    return f"{span['name']}  [{span['run_type']}]  {'  '.join(marks)}"


def ensure_model_price() -> str:
    """Реєструє ціну моделі демо у прайс-листі LangSmith. Ідемпотентно.

    Ціни беруться з .env (долари за 1M токенів) — їх знає той, хто платить за
    шлюз, а не код:

        LLM_PRICE_INPUT=0.19
        LLM_PRICE_OUTPUT=0.51
    """
    model = os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL", "")
    prompt, completion = os.getenv("LLM_PRICE_INPUT"), os.getenv("LLM_PRICE_OUTPUT")
    if not (TRACING and KEY):
        return "трасування вимкнене — ціни не потрібні"
    if not (prompt and completion):
        return (f"ціна для {model} не задана (LLM_PRICE_INPUT/LLM_PRICE_OUTPUT); "
                "публічні моделі LangSmith рахує сам")

    # Свої записи від чужих відрізняє tenant_id: він є лише в тих, що створила
    # ваша організація. Чужі не чіпаємо, свій не дублюємо.
    mine = [e for e in (_get("model-price-map", q=model, limit=50) or [])
            if e.get("tenant_id") and e["name"] == model]
    if mine:
        return f"ціна {model} вже зареєстрована"

    requests.post(f"{API}/api/v1/model-price-map", headers={"x-api-key": KEY}, timeout=20,
                  json={"name": model,
                        "match_pattern": f"^{model}$",
                        "prompt_cost": float(prompt) / 1e6,
                        "completion_cost": float(completion) / 1e6}).raise_for_status()
    return f"ціну {model} зареєстровано: ${prompt}/${completion} за 1M токенів"


if __name__ == "__main__":
    print(ensure_model_price())
