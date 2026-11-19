"""s7 — Reflexion: агент оцінює сам себе, доки не стане достатньо добре.

Потрібен ключ LLM. Запуск:  python s7_reflexion.py

Патерн: generate → critique → improve → repeat. Це «eval усередині пайплайна»:
ті самі критерії, що в s6, але застосовані під час роботи, а не в CI.

Дивитись на:
  • у трасі кожна ітерація — окремий span (generate + critique), видно, як
    оцінка росте і на чому цикл зупинився;
  • критерій зупинки — конкретне число (score ≥ 8), а не «відповідь хороша»;
  • ліміт ітерацій обовʼязковий: цикл без ліміту — це LLM10 Unbounded
    Consumption, тільки повільно і за ваш рахунок;
  • підсумкові метрики: скільки коштувала якість (токени × ітерації).
"""
from printing import print_step, print_table, print_title
from provider import make_llm, text_of
from tracing import Tracer

TASK = ("Напиши відповідь клієнту Ігорю: його замовлення A-1002 (чохол Aura Case, "
        "1250 грн) їде другий тиждень, статус shipped, точної дати доставки немає. "
        "Виведи ЛИШЕ один готовий текст листа, без варіантів і пояснень, до 6 речень.")

CRITIC = (
    "Ти — редактор служби підтримки. Оціни відповідь клієнту за критеріями:\n"
    "1) є вибачення і емпатія; 2) названо конкретний статус і номер замовлення;\n"
    "3) є наступний крок із термінами; 4) без вигаданих фактів і без обіцянок,\n"
    "яких не можна виконати; 5) не довше 6 речень.\n\n"
    "ВІДПОВІДЬ:\n{draft}\n\n"
    "Оцінюй суворо: 9 або 10 — лише якщо виконані ВСІ пʼять критеріїв без жодного "
    "зауваження, зокрема названо конкретний термін наступного кроку.\n"
    "Формат відповіді рівно один рядок: SCORE=<1-10>; FIX=<що виправити, до 20 слів>"
)

IMPROVE = ("Ось відповідь клієнту і зауваження редактора. Перепиши відповідь, "
           "усунувши зауваження. Виведи ЛИШЕ новий текст відповіді.\n\n"
           "ВІДПОВІДЬ:\n{draft}\n\nЗАУВАЖЕННЯ:\n{fix}")

MAX_ITERATIONS = 3
GOOD_ENOUGH = 9   # поріг навмисно високий: з першого разу так майже ніколи не виходить


def parse(verdict: str) -> tuple[int, str]:
    try:
        score = int(verdict.split("SCORE=")[1].split(";")[0].strip())
    except (IndexError, ValueError):
        score = 0
    fix = verdict.split("FIX=")[-1].strip() if "FIX=" in verdict else ""
    return score, fix


if __name__ == "__main__":
    print_title("s7 · Reflexion: generate → critique → improve")

    # У проді критик — дешева модель (flash/mini), генератор — дорога.
    # Тут обидві однакові, щоб демо працювало на безкоштовному тарифі.
    generator, critic = make_llm(), make_llm()
    tracer = Tracer("s7.reflexion")

    draft, history = "", []
    with tracer.span("reflexion_loop", "agent", inp=TASK):
        for i in range(1, MAX_ITERATIONS + 1):
            print_step(f"Ітерація {i}")
            with tracer.span(f"generate#{i}", "llm") as span:
                prompt = TASK if not draft else IMPROVE.format(draft=draft, fix=history[-1][2])
                response = generator.invoke(prompt)
                draft = text_of(response)
                usage = getattr(response, "usage_metadata", None) or {}
                span.attrs.update(in_tok=usage.get("input_tokens", 0),
                                  out_tok=usage.get("output_tokens", 0))
                span.out = draft
            print("  чернетка:\n    " + draft.replace("\n", "\n    ")[:600])

            with tracer.span(f"critique#{i}", "eval") as span:
                verdict_msg = critic.invoke(CRITIC.format(draft=draft))
                verdict = text_of(verdict_msg).strip()
                usage = getattr(verdict_msg, "usage_metadata", None) or {}
                span.attrs.update(in_tok=usage.get("input_tokens", 0),
                                  out_tok=usage.get("output_tokens", 0))
                score, fix = parse(verdict)
                span.attrs["score"] = score
                span.out = verdict
            print(f"\n  критик: SCORE={score} · {fix}")
            history.append((i, score, fix))

            if score >= GOOD_ENOUGH:
                tracer.event("stop", "eval", out=f"score {score} ≥ {GOOD_ENOUGH}")
                print(f"  → зупиняємось: {score} ≥ {GOOD_ENOUGH}")
                break
        else:
            tracer.event("stop", "eval", out=f"вичерпано ліміт {MAX_ITERATIONS} ітерацій")
            print(f"  → зупиняємось за лімітом ітерацій ({MAX_ITERATIONS}), "
                  f"а не за якістю — це нормально і це треба логувати")

    print_step("Траса циклу: кожна ітерація видима окремо")
    tracer.print_tree()

    print_step("Динаміка оцінки і ціна якості")
    print_table([(i, score, fix[:56]) for i, score, fix in history],
          ("ітерація", "score", "зауваження критика"))
    tracer.print_metrics()
    m = tracer.metrics()
    if any(b[1] < a[1] for a, b in zip(history, history[1:])):
        print("\n  Зверніть увагу: оцінка не тільки росла. Критик — теж LLM і теж")
        print("  недетермінований, тому Reflexion не гарантує монотонного покращення;")
        print("  у проді беруть кращу за score чернетку, а не останню.")
    print(f"\n  Ціна: {m['llm_calls']} виклики LLM замість 1 — саме тому ітерацій 2–3,")
    print("  критик на дешевій моделі, а критерій зупинки — число, а не відчуття.")
    print("\nФІНАЛЬНА ВІДПОВІДЬ КЛІЄНТУ:\n")
    print("  " + draft.replace("\n", "\n  "))
