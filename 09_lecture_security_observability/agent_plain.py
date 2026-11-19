"""Той самий ReAct-агент, але без жодного guardrail — лише трейсинг.

Це базова лінія до `agent.py`: цикл «модель → інструмент → модель», обгорнутий
у спани, і більше нічого. Ані before_agent, ані after_tool — між LLM і зовнішнім
світом порожньо. Показує дві речі одразу:

1. Трейсинг — не частина захисту. Спани тут ті самі (`agent.*`, `llm.call#N`,
   `tool.*`), дерево у консолі виглядає так само, лічильники токенів на місці —
   а агент при цьому беззахисний. Спостережуваність каже, ЩО сталося, і нічого
   не забороняє.
2. Скільки коду насправді додають хуки. Порівняйте з `agent.py`: різниця —
   шість `if`, решта однакова.

max_turns залишається: це не guardrail у сенсі OWASP, а елементарний запобіжник
від нескінченного циклу (LLM10 Unbounded Consumption). Без нього демо, яке пішло
не туди, зупиняється тільки на грошах.

Запускати цим агентом небезпечні сценарії (s1_injection) корисно саме тому, що
видно повний ланцюжок атаки в трейсі: жоден крок не заблокований.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from provider import make_llm, text_of
from tracing import Tracer


class BudgetExceeded(RuntimeError):
    """Агент вичерпав ліміт кроків (LLM10)."""


class PlainAgent:
    def __init__(self, name: str, system_prompt: str, tools: list, tracer: Tracer,
                 max_turns: int = 6):
        self.name, self.system_prompt, self.tracer = name, system_prompt, tracer
        self.tools = {t.name: t for t in tools}
        self.max_turns = max_turns
        self.llm = make_llm().bind_tools(tools)

    def run(self, user_text: str) -> str:
        with self.tracer.span(f"agent.{self.name}", "agent", inp=user_text) as root:
            msgs = [SystemMessage(self.system_prompt), HumanMessage(user_text)]
            for turn in range(1, self.max_turns + 1):
                ai_msg: AIMessage = self._call_llm(msgs, turn)
                msgs.append(ai_msg)
                if not ai_msg.tool_calls:
                    answer = text_of(ai_msg)
                    root.out = answer
                    return answer
                for tc in ai_msg.tool_calls:
                    msgs.append(ToolMessage(self._run_tool(tc["name"], tc["args"]),
                                            tool_call_id=tc["id"]))
            # Ліміт вичерпано: краще зупинитись, ніж крутитись далі.
            raise BudgetExceeded(f"{self.name}: вичерпано ліміт {self.max_turns} кроків")

    # ── внутрішнє ───────────────────────────────────────────────────────────
    def _call_llm(self, msgs: list, turn: int) -> AIMessage:
        with self.tracer.span(f"llm.call#{turn}", "llm") as span:
            ai_msg = self.llm.invoke(msgs)
            usage = getattr(ai_msg, "usage_metadata", None) or {}
            span.attrs["in_tok"] = usage.get("input_tokens", 0)
            span.attrs["out_tok"] = usage.get("output_tokens", 0)
            span.out = (text_of(ai_msg) or
                        "→ " + ", ".join(f"{c['name']}({c['args']})" for c in ai_msg.tool_calls))
            return ai_msg

    def _run_tool(self, name: str, args: dict) -> str:
        with self.tracer.span(f"tool.{name}", "tool", inp=str(args)) as span:
            result = str(self.tools[name].invoke(args))
            span.out = result
            return result
