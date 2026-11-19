"""Мінімальний ReAct-агент із трейсінгом та шістьма точками перехоплення.

Циклу тут рівно стільки, скільки треба, щоб побачити місця, куди вішаються
guardrails: before_agent, before_model, after_model, before_tool, after_tool,
after_agent. Кожна — просто `if` у коді нижче, тому видно, що захист живе між
LLM і зовнішнім світом.

Пара model-level хуків — не дублікат agent-level. before_agent бачить лише те,
що написав користувач, а before_model — увесь запит, який реально їде в модель:
системний промпт, історію і результати інструментів; саме там ловиться indirect
injection, що приїхала з даних. Симетрично after_model бачить СИРУ відповідь
моделі, ще з tool_calls, тобто може зупинити небезпечний виклик до того, як той
дійде до before_tool; after_agent же отримує тільки фінальний текст користувачу.

max_turns — це LLM10 Unbounded Consumption: цикл «модель → інструмент → модель»
без ліміту зупиняється тільки тоді, коли скінчаться гроші.

Два чесні розходження з продакшн-фреймворками, щоб не було сюрпризів:

1. Тут пара незалежних колбеків before/after. У LangChain v1 і Microsoft Agent
   Framework переважає інша форма — обгортка навколо виклику, яка сама вирішує,
   чи кликати внутрішнє: wrap_tool_call(request, handler), process(context,
   call_next). Обгортка вміє те, чого пара before/after не вміє принципово:
   заміряти час навколо виклику, перехопити виняток усередині й зробити ретрай
   або фолбек, не виходячи з хука. Ідея та сама, форма інша.
2. Тут немає хуків на помилку. У ADK-плагінів це on_tool_error / on_model_error,
   у LangGraph і MAF воно виходить саме собою з try/except навколо обгортки.
   У нас виняток інструмента просто летить нагору — а саме там живе graceful
   degradation і захист від каскадних відмов (ASI08).

Як ці ж точки виглядають у справжніх фреймворках — у бонусних s8 і s9.
"""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from provider import make_llm, text_of
from tracing import Tracer


class BudgetExceeded(RuntimeError):
    """Агент вичерпав ліміт кроків (LLM10)."""


def prompt_text(msgs: list) -> str:
    """Запит до моделі одним рядком — вхід llm-спана.

    Без цього в трасі у виклику моделі є вихід і токени, але немає входу:
    платформа не показує, ЩО саме бачила модель, коли вирішила кликати
    інструмент. Список повідомлень LangChain сам собою не серіалізується,
    тому зводимо його до «роль: текст» руками.
    """
    def one(m) -> str:
        body = text_of(m) or ", ".join(f"{c['name']}({c['args']})"
                                       for c in getattr(m, "tool_calls", []) or [])
        return f"{m.type}: {body}"
    return "\n".join(one(m) for m in msgs)


class Agent:
    def __init__(self, name: str, system_prompt: str, tools: list, tracer: Tracer,
                 max_turns: int = 6,
                 before_agent: Callable[[str], str | None] | None = None,
                 before_model: Callable[[list], str | None] | None = None,
                 after_model: Callable[[AIMessage], str | None] | None = None,
                 before_tool: Callable[[str, dict], str | None] | None = None,
                 after_tool: Callable[[str, dict, str], str] | None = None,
                 after_agent: Callable[[str], str] | None = None):
        self.name, self.system_prompt, self.tracer = name, system_prompt, tracer
        self.tools = {t.name: t for t in tools}
        self.max_turns = max_turns
        self.before_agent, self.before_tool = before_agent, before_tool
        self.before_model, self.after_model = before_model, after_model
        self.after_tool, self.after_agent = after_tool, after_agent
        self.llm = make_llm().bind_tools(tools)

    def run(self, user_text: str) -> str:
        with self.tracer.span(f"agent.{self.name}", "agent", inp=user_text) as root:
            # before_agent: Патерни ін'єкції в тексті користувача.
            if self.before_agent and (blocked := self.before_agent(user_text)):
                root.out = blocked
                return blocked

            msgs = [SystemMessage(self.system_prompt), HumanMessage(user_text)]
            for turn in range(1, self.max_turns + 1):
                ai_msg: AIMessage = self._call_llm(msgs, turn)
                msgs.append(ai_msg)
                if not ai_msg.tool_calls:
                    answer = text_of(ai_msg)
                    # after_agent: Вихідний guardrail: PII, секрети, заборонений контент.
                    if self.after_agent:
                        answer = self.after_agent(answer)
                    root.out = answer
                    return answer
                for tc in ai_msg.tool_calls:
                    msgs.append(ToolMessage(self._run_tool(tc["name"], tc["args"]),
                                            tool_call_id=tc["id"]))
            # Ліміт вичерпано: краще зупинитись, ніж крутитись далі.
            raise BudgetExceeded(f"{self.name}: вичерпано ліміт {self.max_turns} кроків")

    # ── внутрішнє ───────────────────────────────────────────────────────────
    def _call_llm(self, msgs: list, turn: int) -> AIMessage:
        with self.tracer.span(f"llm.call#{turn}", "llm", inp=prompt_text(msgs)) as span:
            # before_model: Кеш, policy-guardrail і аварійне вимкнення.
            if self.before_model and (intercepted := self.before_model(msgs)):
                span.out, span.attrs["short_circuit"] = intercepted, True
                return AIMessage(intercepted)
            ai_msg = self.llm.invoke(msgs)
            usage = getattr(ai_msg, "usage_metadata", None) or {}
            span.attrs["in_tok"] = usage.get("input_tokens", 0)
            span.attrs["out_tok"] = usage.get("output_tokens", 0)
            span.out = (text_of(ai_msg) or
                        "→ " + ", ".join(f"{c['name']}({c['args']})" for c in ai_msg.tool_calls))
            # after_model: Заборонений tool_call або PII: сиру відповідь видно тільки тут.
            if self.after_model and (rewritten := self.after_model(ai_msg)):
                span.out, span.attrs["rewritten"] = rewritten, True
                return AIMessage(rewritten)
            return ai_msg

    def _run_tool(self, name: str, args: dict) -> str:
        with self.tracer.span(f"tool.{name}", "tool", inp=str(args)) as span:
            # before_tool: Allowlist, ліміт суми й кількості викликів, approval людини.
            if self.before_tool and (denied := self.before_tool(name, args)):
                span.out, span.attrs["blocked"] = denied, True
                return denied
            result = str(self.tools[name].invoke(args)) if name in self.tools else \
                f"Інструмент {name} невідомий."
            # after_tool: Санітизація недовірених даних: indirect injection із тікета.
            if self.after_tool:
                sanitized = self.after_tool(name, args, result)
                if sanitized != result:
                    span.attrs["sanitized"] = True
                result = sanitized
            span.out = result
            return result
