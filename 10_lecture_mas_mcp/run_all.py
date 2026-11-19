"""Прогін усього практикуму по порядку: кроки 1-7 плюс тести.

Потрібен, щоб перевірити середовище перед заняттям або зібрати вивід для
здачі:

    python run_all.py                # усе
    python run_all.py --offline      # лише те, що працює без ключа LLM
    python run_all.py 3 5            # вибіркові кроки

Вивід кожного кроку йде в консоль як є. Кроки з LLM недетерміновані —
формулювання відповідей щоразу різні, набір викликів tools зазвичай той самий.
"""
from __future__ import annotations

import subprocess
import sys

STEPS: list[tuple[str, str, bool]] = [
    # (номер, команда, чи потрібен ключ LLM)
    ("1", "step1_mcp_server.py", False),
    ("t", "-m pytest -q", False),
    ("2", "step2_mcp_agent.py", True),
    ("3", "step3_mas_supervisor.py", True),
    ("4", "step4_guardrails.py", False),
    ("5", "step5_secured_mas.py", True),
    ("6", "step6_hitl.py", True),
    ("7", "step7_tracing.py", True),
]


def main() -> int:
    args = sys.argv[1:]
    offline = "--offline" in args
    wanted = {a for a in args if not a.startswith("--")}

    for number, command, needs_llm in STEPS:
        if wanted and number not in wanted:
            continue
        if offline and needs_llm:
            continue
        print(f"\n\n{'#' * 78}\n# крок {number}: python {command}\n{'#' * 78}")
        result = subprocess.run([sys.executable, *command.split()])
        if result.returncode != 0:
            print(f"\nКрок {number} завершився з кодом {result.returncode}.")
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
