"""Друк у консоль: заголовки, кроки і таблиці для демо лекції 9.

Домен усіх скриптів один — агент підтримки інтернет-магазину «ShopBot».
Тут немає нічого, крім оформлення виводу: провайдер моделі живе у provider.py,
трасувальник — у tracing.py, guardrails — у guards.py.
"""
from __future__ import annotations


# ── дрібні друкарські помічники, щоб читалось краще. Не відноситься до основного розділу матеріалу
def print_title(text: str) -> None:
    print(f"\n\033[1m{'═' * 78}\n{text}\n{'═' * 78}\033[0m")


def print_step(text: str) -> None:
    print(f"\n── {text} " + "─" * max(0, 74 - len(text)))


def print_table(rows: list[tuple], headers: tuple) -> None:
    cols = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    line = "  ".join(str(h).ljust(cols[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * c for c in cols))
    for r in rows:
        print("  ".join(str(v).ljust(cols[i]) for i, v in enumerate(r)))
