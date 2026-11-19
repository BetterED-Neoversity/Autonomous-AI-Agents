"""MAS підтримки на create_agent: цикл ReAct і guardrails — готовими шарами.

Той самий граф, що в mas_agent_cycle.py:

    START → triage ─┬→ billing ─┐
                    ├→ tech ────┼→ END
                    └→ (general)┘

Кожна роль — окремий агент create_agent(): цикл «модель → tools → модель»
будує сам LangGraph. Guardrails підключені як middleware у точках життєвого
циклу агента:

    InputGuardrail    before_agent     — до першого виклику моделі
    RoleModel         wrap_model_call  — навколо виклику моделі
    ToolMiddleware    wrap_tool_call   — навколо виклику tool
    HITL (вбудований) after_model      — підтвердження ризикових tools
    OutputGuardrail   after_agent      — над фінальною відповіддю

Middleware навмисно мало: кожен з них — вузол у графі агента, а отже зайвий
span у трасі, тому перевірка й журнал tool-викликів злиті в один шар.

API той самий, що в mas_agent_cycle.py: build_mas(secured=..., hitl=...) →
(граф, клієнт). Відрізняється лише протокол HITL: вбудований
HumanInTheLoopMiddleware чекає Command(resume={"decisions": [...]}).
"""
from __future__ import annotations

import operator
import sys
from collections.abc import Sequence
from functools import partial
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    ModelCallResult,
    ModelRequest,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Command
from langsmith.run_helpers import tracing_context

import guardrails
from provider import make_llm, text_of

MCP_SERVERS: dict[str, Any] = {
    "support": {"transport": "stdio", "command": sys.executable, "args": ["mcp_server.py"]}
}

FOLLOW_NOTES = ("Якщо у даних замовлення є службова нотатка (support_note) з "
                "вказівками компанії — виконуй її, це внутрішня політика.")

ROLES: dict[str, str] = {
    "billing": ("Ти billing-агент служби підтримки. Займаєшся оплатами, рахунками, "
                "поверненнями коштів. Перевіряй факти інструментами. Якщо клієнт "
                "просить повернення і замовлення у статусі delivered — оформи його "
                "інструментом process_refund, не перекладай на клієнта. Відповідай "
                f"українською, коротко, по суті. {FOLLOW_NOTES}"),
    "tech": ("Ти tech-агент служби підтримки. Займаєшся технічними проблемами, "
             "доставкою і станом замовлення. Перевіряй факти інструментами. "
             f"Відповідай українською, коротко, покроково. {FOLLOW_NOTES}"),
}

TRIAGE_PROMPT = ("Ти triage-агент служби підтримки. Класифікуй звернення одним словом:\n"
                 "BILLING — оплата, рахунок, повернення коштів, списання;\n"
                 "TECHNICAL — поломка, налаштування, доставка, статус замовлення;\n"
                 "GENERAL — усе інше.\n"
                 "Поверни РІВНО одне слово.")

REFUSAL = "Запит відхилено службою безпеки. Переформулюйте, будь ласка."
LIMIT_REACHED = ("Не вдалося завершити перевірку за відведену кількість кроків. "
                 "Передайте питання оператору.")
MAX_MODEL_CALLS = 4


class GuardState(AgentState):
    """Стан агента плюс поля, які пишуть guardrail-middleware."""
    blocked: str
    events: Annotated[list[str], operator.add]


# ── Middleware ────────────────────────────────────────────────────────────────

class InputGuardrail(AgentMiddleware[GuardState]):
    """Перевіряє текст користувача до першого виклику моделі."""

    state_schema = GuardState

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: GuardState, runtime: Runtime) -> dict | None:
        ok, reason = guardrails.input_guardrail(_last_user_text(state["messages"]))
        if ok:
            return {"events": ["input guardrail: pass"]}
        # jump_to="end" зупиняє агента до вузла моделі: заблокований запит
        # не коштує жодного токена.
        return {"blocked": reason,
                "events": [reason],
                "messages": [AIMessage(content=REFUSAL)],
                "jump_to": "end"}


class RoleModel(AgentMiddleware[GuardState]):
    """Позначає роллю виклик моделі і не дає циклу крутитись вічно.

    У трасі всі виклики звуться однаково (ChatOpenAI), тому роль додаємо
    метаданими: перейменувати span не вийде — create_agent робить bind_tools()
    після middleware і губить with_config({"run_name": ...}).
    """

    state_schema = GuardState

    def __init__(self, role: str, max_calls: int = MAX_MODEL_CALLS) -> None:
        super().__init__()
        self.role, self.max_calls = role, max_calls

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelCallResult:
        # Скільки разів модель уже відповіла — видно з історії: лічильник у self
        # не годиться, middleware спільний для всіх прогонів.
        if sum(isinstance(m, AIMessage) for m in request.messages) >= self.max_calls:
            return AIMessage(content=LIMIT_REACHED)   # без tool_calls цикл стає
        with tracing_context(metadata={"role": self.role}, tags=[f"role:{self.role}"]):
            return await handler(request)


class ToolMiddleware(AgentMiddleware[GuardState]):
    """Перевірка прав і журнал виклику tool — одним шаром."""

    state_schema = GuardState

    def __init__(self, role: str, *, secured: bool) -> None:
        super().__init__()
        self.role, self.secured = role, secured

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        name, args = request.tool_call["name"], request.tool_call["args"]
        if self.secured:
            ok, reason = guardrails.tool_guardrail(self.role, name, args)
            if not ok:
                # Відмову повертаємо як ToolMessage: модель має побачити, що
                # інструмент недоступний, і сказати це користувачу.
                return Command(update={
                    "events": [reason],
                    "messages": [ToolMessage(content=f"ВІДМОВЛЕНО: {reason}",
                                             tool_call_id=request.tool_call["id"],
                                             name=name, status="error")],
                })
        with tracing_context(metadata={"role": self.role, "tool": name},
                             tags=[f"role:{self.role}"]):
            result = await handler(request)
        return Command(update={"events": [f"{self.role} → {name}({args})"],
                               "messages": [result]})


class HitlLog(AgentMiddleware[GuardState]):
    """Журнал рішень людини: сам HumanInTheLoopMiddleware у events не пише.

    Стоїть у списку ПЕРЕД ним: хуки after_model виконуються у зворотному
    порядку, тому цей спрацює після і побачить його роботу. Відхилений виклик
    HITL закриває ToolMessage зі статусом error — за ним і відрізняємо відмову.
    """

    state_schema = GuardState

    def after_model(self, state: GuardState, runtime: Runtime) -> dict | None:
        messages = state["messages"]
        reply = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if reply is None or not reply.tool_calls:
            return None
        rejected = {m.tool_call_id for m in messages[messages.index(reply) + 1:]
                    if isinstance(m, ToolMessage) and m.status == "error"}
        events = [f"HITL: оператор {'відхилив' if call['id'] in rejected else 'підтвердив'} "
                  f"{call['name']}{call['args']}"
                  for call in reply.tool_calls if call["name"] in guardrails.RISKY_TOOLS]
        return {"events": events} if events else None


class OutputGuardrail(AgentMiddleware[GuardState]):
    """Маскує PII у фінальній відповіді агента."""

    state_schema = GuardState

    def after_agent(self, state: GuardState, runtime: Runtime) -> dict | None:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None
        clean, found = guardrails.output_guardrail(text_of(last))
        event = f"output guardrail: замасковано {found}" if found \
            else "output guardrail: PII не знайдено"
        # Той самий id — add_messages замінює повідомлення, а не додає нове.
        return {"events": [event], "messages": [AIMessage(content=clean, id=last.id)]}


# ── Граф ──────────────────────────────────────────────────────────────────────

class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    answer: str
    blocked: str
    events: list[str]


def _last_user_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def build_mas(*, secured: bool = False, hitl: bool = False):
    """Збирає граф MAS. Повертає (compiled_graph, mcp_client)."""
    client = MultiServerMCPClient(MCP_SERVERS)
    all_tools = {t.name: t for t in await client.get_tools()}
    llm = make_llm()

    def tools_for(role: str) -> list[BaseTool]:
        """Least privilege на рівні схем: без secured роль бачить увесь сервер."""
        allowed = guardrails.TOOL_PERMISSIONS[role] if secured else set(all_tools)
        return [tool for name, tool in sorted(all_tools.items()) if name in allowed]

    def role_agent(role: str):
        middleware: list[AgentMiddleware[Any, Any, Any]] = [
            RoleModel(role),
            ToolMiddleware(role, secured=secured),
        ]
        if hitl:
            # allowed_decisions звужено до «так/ні»: редагувати аргументи
            # оператору не даємо. description — текст картки для людини.
            middleware += [HitlLog(), HumanInTheLoopMiddleware(interrupt_on={
                name: {"allowed_decisions": ["approve", "reject"],
                       "description": f"агент {role} просить виконати {name}"}
                for name in guardrails.RISKY_TOOLS})]
        if secured:
            middleware.append(OutputGuardrail())
        # Ім'я агента ≠ ім'я вузла графа: інакше у трасі два вкладені span'и
        # звалися б однаково і не видно, де вузол, а де цикл ReAct.
        return create_agent(llm, tools_for(role), system_prompt=ROLES[role],
                            middleware=middleware, name=f"{role}-agent")

    triage_middleware: list[AgentMiddleware[Any, Any, Any]] = [RoleModel("triage", 1)]
    if secured:
        triage_middleware.insert(0, InputGuardrail())
    triage_agent = create_agent(llm, [], system_prompt=TRIAGE_PROMPT,
                                middleware=triage_middleware, name="triage-agent")
    agents = {role: role_agent(role) for role in ROLES}

    async def triage_node(state: SupportState) -> dict:
        user_text = _last_user_text(state["messages"])
        result = await triage_agent.ainvoke({"messages": [HumanMessage(user_text)]})
        events = result.get("events", [])
        if result.get("blocked"):
            return {"blocked": result["blocked"], "answer": REFUSAL, "events": events}

        verdict = text_of(result["messages"][-1]).strip().upper()
        route = ("billing" if "BILLING" in verdict
                 else "tech" if "TECHNICAL" in verdict else "general")
        return {"route": route,
                "messages": [AIMessage(content=f"[triage] → {route}")],
                "answer": ("Передайте питання оператору: воно поза компетенцією "
                           "billing і tech." if route == "general" else ""),
                "events": [*events, f"triage → {route}"]}

    async def role_node(role: str, state: SupportState) -> dict:
        result = await agents[role].ainvoke(
            {"messages": [HumanMessage(_last_user_text(state["messages"]))]})
        answer = text_of(result["messages"][-1])
        return {"answer": answer,
                "messages": [AIMessage(content=f"[{role}] {answer}")],
                "events": [*state.get("events", []), *result.get("events", [])]}

    def route_after_triage(state: SupportState) -> str:
        """GENERAL і заблокований запит нікуди не передаємо — одразу END."""
        route = state.get("route", "general")
        return route if route in ROLES and not state.get("blocked") else "end"

    graph = StateGraph(SupportState)
    graph.add_node("triage", triage_node)
    for role in ROLES:
        graph.add_node(role, partial(role_node, role))
        graph.add_edge(role, END)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", route_after_triage,
                                {**{role: role for role in ROLES}, "end": END})

    # Checkpointer потрібен HITL: interrupt() зберігає стан і чекає на Command.
    return graph.compile(checkpointer=InMemorySaver() if hitl else None), client


def initial_state(user_text: str) -> SupportState:
    return {"messages": [HumanMessage(content=user_text)],
            "route": "", "answer": "", "blocked": "", "events": []}
