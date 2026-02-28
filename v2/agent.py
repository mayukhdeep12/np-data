"""
agent.py — NEODesk LangGraph ReAct Agent  (with Guardrails)
============================================================

Graph flow with three guardrail layers:

    [User Input]
         │
         ▼
    ┌──────────────────────────────────────┐
    │  LAYER 1: INPUT GUARDRAIL NODE       │  ← Prompt injection, credentials,
    │  (deterministic regex + LLM check)   │    off-topic / harmful content
    └────────────┬─────────────────────────┘
                 │                   │ BLOCKED
                 │ PASSED            ▼
                 │              [__end__]  ← Block message returned to user
                 ▼
    ┌─────────────────────────────┐
    │  AGENT NODE  (LLM Reasons)  │  ◄──────────────────────┐
    └────────────┬────────────────┘                          │
                 │                                           │
      ┌──────────▼──────────┐                               │
      │  Conditional Edge   │                               │
      │  should_use_tool()  │                               │
      └──────────┬──────────┘                               │
                 │                                           │
        ┌────────┴──────────┐                               │
        ▼                   ▼                               │
  tool_calls?          No tool_calls?                       │
        │                   │                               │
        ▼                   ▼                               │
  ┌──────────────┐    ┌──────────────────────────────────┐  │
  │  LAYER 2:    │    │  LAYER 3: OUTPUT GUARDRAIL NODE   │  │
  │  TOOL        │    │  (PII scrubbing + LLM compliance) │  │
  │  GUARDRAIL   │    └──────────────┬───────────────────┘  │
  │  (per-call   │                   │                       │
  │  rule check) │              [__end__]                    │
  └──────┬───────┘         Final answer delivered            │
    OK   │  BLOCKED                                          │
         │     │                                             │
         ▼     ▼                                             │
  ┌──────────────┐                                           │
  │  TOOL NODE   │                                           │
  │  Execute MCP │                                           │
  │  or built-in │                                           │
  └──────┬───────┘                                           │
         │  ToolMessage(s) appended to state                 │
         └───────────────────────────────────────────────────┘
                             Loop back
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Sequence, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from guardrails import InputGuardrail, ToolGuardrail, OutputGuardrail, GuardrailBlock


# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN TOOLS  (always available — no MCP connection needed)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_loan_rates() -> str:
    """Get current loan and mortgage interest rates offered by NEODesk bank."""
    return (
        "Current NEODesk Loan Rates:\n"
        "  Personal Loan   : 6.99% – 18.99%\n"
        "  Home Mortgage   : 5.25% –  7.50%\n"
        "  Auto Loan       : 4.99% – 12.99%\n"
        "  Student Loan    : 3.75% –  8.50%\n"
        "  Business Loan   : 7.25% – 19.99%\n"
        "Rates subject to credit approval. Valid today."
    )


@tool
def get_banking_faq(topic: str) -> str:
    """
    Answer common banking FAQs.

    Args:
        topic: One of — fees, hours, contact, security, limits
    """
    answers = {
        "fees": (
            "NEODesk charges zero monthly fees on checking accounts. "
            "Savings earns 4.5% APY. Domestic wire: $15. International wire: $30."
        ),
        "hours": (
            "Branches: Mon–Fri 9 am–5 pm, Sat 10 am–2 pm. "
            "Online banking and this assistant are available 24/7."
        ),
        "contact": (
            "Customer support: 1-800-NEO-DESK (Mon–Fri 8 am–8 pm ET). "
            "Fraud hotline: 1-800-NEO-SAFE (24/7)."
        ),
        "security": (
            "All accounts are protected by 256-bit TLS encryption, "
            "two-factor authentication, and real-time fraud monitoring."
        ),
        "limits": (
            "Daily ATM withdrawal: $1,000. "
            "Daily debit purchases: $5,000. "
            "Wire transfer: $25,000/day."
        ),
    }
    for key, answer in answers.items():
        if key in topic.lower():
            return answer
    return (
        f"I don't have specific FAQ info for '{topic}'. "
        "Please call 1-800-NEO-DESK for assistance."
    )


BUILTIN_TOOLS = [get_loan_rates, get_banking_faq]


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are NEODesk, a professional AI banking assistant.

────────────────────────────────────────────────
AVAILABLE TOOLS
────────────────────────────────────────────────

MCP Banking Tools  (live account data):
  • get_account_balance      — real-time balance for ACC001, ACC002, ACC003
  • get_all_accounts         — summary of all linked accounts
  • get_card_status          — card details for CARD001, CARD002
  • update_card_status       — block / unblock / freeze a card
  • detect_fraud             — fraud risk score + flagged transactions
  • get_transaction_history  — transactions with optional category filter
  • transfer_funds           — move money between two accounts
  • get_spending_analytics   — category-wise spending breakdown
  • get_beneficiaries        — list all saved payees
  • add_beneficiary          — register a new payee
  • get_mini_statement       — last 5 transactions + current balance

Built-in Tools  (static knowledge):
  • get_loan_rates            — current interest rates
  • get_banking_faq           — fees, hours, contact, security, daily limits

────────────────────────────────────────────────
DECISION RULES  (apply at every step of reasoning)
────────────────────────────────────────────────

1. Query involves accounts, cards, transactions, fraud, or transfers
   → ALWAYS call the relevant MCP tool. Never guess balances or invent data.

2. Query is about loan rates or general banking policies / FAQs
   → Use the built-in tools.

3. Query is conversational, conceptual, or answerable from general knowledge
   → Answer directly. Do NOT call any tool.

4. After receiving a tool result, reason over it:
   → Still need more data? Call another tool.
   → Have everything needed? Compose the final answer and stop.

────────────────────────────────────────────────
RESPONSE GUIDELINES
────────────────────────────────────────────────
- Format all currency as $X,XXX.XX
- Be concise and professional
- Never request passwords, PINs, or full card numbers
- For complex financial planning, recommend a certified financial advisor
"""


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:         Annotated[Sequence[BaseMessage], add_messages]
    # Guardrail metadata — not shown to the user, used internally
    guardrail_blocked: Optional[bool]    # True if any layer blocked
    guardrail_code:    Optional[str]     # reason_code from GuardrailBlock


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(groq_api_key: str, extra_tools: list = None):
    """
    Compile the NEODesk ReAct LangGraph agent with three guardrail layers.

    Args:
        groq_api_key : Groq API key
        extra_tools  : MCP tools loaded by MCPManager (pass [] or None if not connected)

    Returns:
        Compiled StateGraph with MemorySaver checkpointer.
    """

    all_tools = BUILTIN_TOOLS + (extra_tools or [])
    tool_map  = {t.name: t for t in all_tools}

    # Guardrail instances (shared across all graph invocations)
    input_guard  = InputGuardrail(groq_api_key,  enable_llm_check=True)
    tool_guard   = ToolGuardrail()
    output_guard = OutputGuardrail(groq_api_key, enable_llm_check=True)

    # LLM with tool schemas
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        streaming=True,
    ).bind_tools(all_tools)

    # Plain LLM — fallback when tool-call generation fails
    plain_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        streaming=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # NODE: input_guardrail
    # Extracts the latest user message and runs Layer 1 checks.
    # On block → injects an AIMessage with the block text so the user sees it.
    # On pass  → leaves state unchanged.
    # ══════════════════════════════════════════════════════════════════════════
    async def input_guardrail_node(state: AgentState) -> dict:
        # Find the most recent human message
        user_text = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_text = msg.content if isinstance(msg.content, str) else ""
                break

        if not user_text:
            return {"guardrail_blocked": False, "guardrail_code": None}

        block: Optional[GuardrailBlock] = input_guard.check(user_text)

        if block:
            return {
                "messages":          [AIMessage(content=block.block_message)],
                "guardrail_blocked": True,
                "guardrail_code":    block.reason_code,
            }

        return {"guardrail_blocked": False, "guardrail_code": None}

    # ══════════════════════════════════════════════════════════════════════════
    # EDGE: after_input_guardrail
    # Route to "__end__" if blocked, else to "agent".
    # ══════════════════════════════════════════════════════════════════════════
    def after_input_guardrail(state: AgentState) -> Literal["agent", "__end__"]:
        if state.get("guardrail_blocked"):
            return "__end__"
        return "agent"

    # ══════════════════════════════════════════════════════════════════════════
    # NODE: agent
    # ══════════════════════════════════════════════════════════════════════════
    async def agent_node(state: AgentState) -> dict:
        conversation = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        try:
            response = await llm.ainvoke(conversation)
            return {"messages": [response]}

        except Exception as exc:
            if "failed_generation" in str(exc) or "Failed to call a function" in str(exc):
                fallback_msgs = conversation + [
                    SystemMessage(content="Answer the user directly without calling any tool.")
                ]
                response = await plain_llm.ainvoke(fallback_msgs)
                return {"messages": [response]}
            raise

    # ══════════════════════════════════════════════════════════════════════════
    # EDGE: should_use_tool
    # ══════════════════════════════════════════════════════════════════════════
    def should_use_tool(
        state: AgentState,
    ) -> Literal["call_tools", "output_guardrail"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tools"
        return "output_guardrail"

    # ══════════════════════════════════════════════════════════════════════════
    # NODE: call_tools  (with Layer 2 guardrail applied per tool call)
    # ══════════════════════════════════════════════════════════════════════════
    async def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_results = []

        for tool_call in last_message.tool_calls:
            name    = tool_call["name"]
            args    = tool_call["args"]
            call_id = tool_call["id"]

            # ── Layer 2: Tool guardrail ────────────────────────────────────────
            block: Optional[GuardrailBlock] = tool_guard.check(name, args)
            if block:
                # Return the block message as a ToolMessage so the agent can
                # relay it naturally to the user in its next reasoning step.
                tool_results.append(
                    ToolMessage(
                        content=f"[GUARDRAIL BLOCKED] {block.block_message}",
                        tool_call_id=call_id,
                    )
                )
                continue  # Skip actual tool execution

            # ── Execute the tool ───────────────────────────────────────────────
            callable_tool = tool_map.get(name)
            if callable_tool is None:
                result = f"Error: Tool '{name}' is not available."
            else:
                try:
                    result = await callable_tool.ainvoke(args)
                except Exception as exc:
                    result = f"Tool '{name}' raised an error: {exc}"

            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=call_id)
            )

        return {"messages": tool_results}

    # ══════════════════════════════════════════════════════════════════════════
    # NODE: output_guardrail
    # Layer 3 — scrubs PII and runs compliance check on the final answer.
    # ══════════════════════════════════════════════════════════════════════════
    async def output_guardrail_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]

        # Only process AI (text) messages — tool messages are internal
        if not isinstance(last_message, AIMessage):
            return {}

        response_text = last_message.content
        if not isinstance(response_text, str):
            return {}

        cleaned_text, block = output_guard.check(response_text)

        if block:
            # Replace response entirely with the compliance-safe fallback
            return {
                "messages":          [AIMessage(content=block.block_message)],
                "guardrail_blocked": True,
                "guardrail_code":    block.reason_code,
            }

        if cleaned_text != response_text:
            # PII was scrubbed — replace the message with the sanitised version
            return {
                "messages": [AIMessage(content=cleaned_text)],
            }

        return {}  # Nothing changed

    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH ASSEMBLY
    # ══════════════════════════════════════════════════════════════════════════

    graph = StateGraph(AgentState)

    graph.add_node("input_guardrail",  input_guardrail_node)
    graph.add_node("agent",            agent_node)
    graph.add_node("call_tools",       tool_node)
    graph.add_node("output_guardrail", output_guardrail_node)

    # Entry point → input guardrail (Layer 1)
    graph.set_entry_point("input_guardrail")

    # After input guardrail → blocked? end : agent
    graph.add_conditional_edges(
        source="input_guardrail",
        path=after_input_guardrail,
        path_map={
            "agent":    "agent",
            "__end__":  END,
        },
    )

    # After agent → needs tool? call_tools : output_guardrail (Layer 3)
    graph.add_conditional_edges(
        source="agent",
        path=should_use_tool,
        path_map={
            "call_tools":       "call_tools",
            "output_guardrail": "output_guardrail",
        },
    )

    # After tools (Layer 2 applied inside) → loop back to agent
    graph.add_edge("call_tools", "agent")

    # After output guardrail → always end
    graph.add_edge("output_guardrail", END)

    return graph.compile(checkpointer=MemorySaver())