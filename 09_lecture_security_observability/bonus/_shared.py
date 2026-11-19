"""Доступ до спільних модулів основної лінії (на рівень вище).

Бонусні скрипти лежать у `bonus/`, а агент, guardrails і домен — у батьківській
папці: `agent.py`, `guards.py`, `shop_tools.py`, `provider.py`, `tracing.py`,
`printing.py`, `evals_gate.py` плюс самі скрипти `s3_secured_agent.py`
і `s6_evals.py`, з яких бонуси беруть захищеного агента й golden dataset.

Сенс саме в тому, щоб нічого не копіювати: бонус доводить свою тезу тільки
якщо агент у ньому той самий, що в основній лінії, до останнього рядка.

Імпортується першим рядком у кожному бонусному скрипті:

    import _shared  # noqa: F401  — додає батьківську папку в sys.path
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
