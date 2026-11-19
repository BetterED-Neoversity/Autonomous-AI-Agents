"""Спільна для s6 і s6b частина evals: пороги, baseline і гейт CI/CD.

Метрики можна рахувати руками (s6) або через deepeval (s6b), а рішення
«деплоїмо чи ні» приймається однаково:

    метрика нижча за поріг          → блокуємо (система працює погано);
    метрика просіла відносно baseline → блокуємо (стало гірше, ніж було).

Друге без першого не працює: абсолютне число ні про що не каже, поки немає
з чим порівняти.
"""
from __future__ import annotations

import json
import pathlib
import sys

from printing import print_table

MAX_DROP = 0.05          # наскільки метриці дозволено просісти відносно baseline


def check_gate(metrics: dict[str, float], thresholds: dict[str, float],
               baseline_path: pathlib.Path, max_drop: float = MAX_DROP) -> list[str]:
    """Друкує таблицю «зараз / поріг / baseline / Δ» і повертає список провалів."""
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    rows, failed = [], []
    for name, value in metrics.items():
        base = baseline.get(name)
        verdict = "OK"
        if value < thresholds[name]:
            verdict = "НИЖЧЕ ПОРОГА"
            failed.append(f"{name} {value} < {thresholds[name]}")
        elif base is not None and value < base - max_drop:
            verdict = "РЕГРЕСІЯ"
            failed.append(f"{name} просів на {base - value:.3f}")
        rows.append((name, value, thresholds[name], base if base is not None else "—",
                     "—" if base is None else f"{value - base:+.3f}", verdict))
    print_table(rows, ("метрика", "зараз", "поріг", "baseline", "Δ", "вердикт"))

    if not baseline_path.exists():
        baseline_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"\n  {baseline_path} створено — наступні прогони порівнюються з ним.")
    return failed


def exit_gate(failed: list[str], script: str) -> None:
    """Код повернення скрипта — це і є гейт CI/CD, який блокує merge."""
    print(f'''  # .github/workflows/evals.yml
  - run: python {script}     # ненульовий код завершення блокує merge''')
    if failed:
        print("\n  ❌ ДЕПЛОЙ ЗАБЛОКОВАНО:")
        for f in failed:
            print(f"     • {f}")
        sys.exit(1)
    print("\n  ✅ Усі метрики в межах — деплой дозволено.")
