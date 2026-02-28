"""
agent.py — NEODesk LangGraph ReAct Agent
=========================================

Implements the full ReAct (Reason + Act) loop:

    [User Input]
         │
         ▼
    ┌─────────────────────────────┐
    │  AGENT NODE  (LLM Reasons)  │  ◄──────────────────────┐
    │                             │                          │
    │  Reads: full message state  │                          │
    │  + system prompt with all   │                          │
    │    tool descriptions        │                          │
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
  ┌──────────────┐      [__end__]                           │
  │  TOOL NODE   │   Final answer                           │
  │  Execute MCP │   delivered                              │
  │  or built-in │                                          │
  └──────┬───────┘                                          │
         │  ToolMessage(s) appended to state                │
         └──────────────────────────────────────────────────┘
                        Loop back

Decision logic lives entirely inside the LLM — it reads tool
descriptions from its bound tools list and decides at every step.
"""

import asyncio
from typing import Annotated, Sequence, Literal
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
# Tells the LLM exactly what tools exist and when to use each one.
# The LLM uses this to decide: call a tool OR reply directly.
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
# Single source of truth shared between every node in the graph.
# `messages` grows with each step: user input → AI response → tool results → …
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(groq_api_key: str, extra_tools: list = None):
    """
    Compile the NEODesk ReAct LangGraph agent.

    Args:
        groq_api_key : Groq API key
        extra_tools  : MCP tools loaded by MCPManager (pass [] or None if not connected)

    Returns:
        Compiled StateGraph with MemorySaver checkpointer.
        Remembers full conversation history per thread_id.
    """

    # All tools available to the LLM this session
    all_tools = BUILTIN_TOOLS + (extra_tools or [])
    tool_map  = {t.name: t for t in all_tools}

    # LLM with tool schemas injected — this is how the LLM "sees" and selects tools
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        streaming=True,
    ).bind_tools(all_tools)

    # Plain LLM without tools — used as fallback if tool-call generation fails
    plain_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        streaming=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — AGENT NODE: LLM Reasoning
    #
    # The LLM receives:
    #   - System prompt (tool descriptions + decision rules)
    #   - Full conversation history (user messages + previous tool results)
    #
    # The LLM outputs one of:
    #   (a) AIMessage WITH tool_calls  → conditional edge routes to TOOL NODE
    #   (b) AIMessage WITHOUT tool_calls → conditional edge routes to END
    # ══════════════════════════════════════════════════════════════════════════
    async def agent_node(state: AgentState) -> dict:
        conversation = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        try:
            response = await llm.ainvoke(conversation)
            return {"messages": [response]}

        except Exception as exc:
            # Groq occasionally fails to serialise a malformed tool call JSON.
            # Fall back to a plain response so the user always gets an answer.
            if "failed_generation" in str(exc) or "Failed to call a function" in str(exc):
                fallback_msgs = conversation + [
                    SystemMessage(
                        content="Answer the user directly without calling any tool."
                    )
                ]
                response = await plain_llm.ainvoke(fallback_msgs)
                return {"messages": [response]}
            raise

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — CONDITIONAL EDGE: Does the LLM want to use a tool?
    #
    # Reads the last message from the agent node:
    #   • tool_calls present  →  route to "call_tools"  (tool needed)
    #   • no tool_calls       →  route to END           (final answer ready)
    #
    # The LLM made this decision — this function just reads its output.
    # ══════════════════════════════════════════════════════════════════════════
    def should_use_tool(state: AgentState) -> Literal["call_tools", "__end__"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tools"
        return "__end__"

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — TOOL NODE: Execute the tool(s) the LLM selected
    #
    # For every tool_call in the last AIMessage:
    #   1. Look up the callable in tool_map
    #   2. Call ainvoke() — works for both async MCP tools and sync built-ins
    #   3. Wrap the result in a ToolMessage and append to state
    #
    # After this node, flow returns to agent_node (STEP 5 — Iteration).
    # ══════════════════════════════════════════════════════════════════════════
    async def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_results = []

        for tool_call in last_message.tool_calls:
            name     = tool_call["name"]
            args     = tool_call["args"]
            call_id  = tool_call["id"]

            callable_tool = tool_map.get(name)
            if callable_tool is None:
                result = f"Error: Tool '{name}' is not available."
            else:
                try:
                    # ainvoke handles both async (MCP) and sync (built-in) tools
                    result = await callable_tool.ainvoke(args)
                except Exception as exc:
                    result = f"Tool '{name}' raised an error: {exc}"

            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=call_id)
            )

        # Tool results are appended to state → agent_node will read them next
        return {"messages": tool_results}

    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH ASSEMBLY
    # ══════════════════════════════════════════════════════════════════════════

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("agent", agent_node)        # STEP 2: LLM reasons
    graph.add_node("call_tools", tool_node)    # STEP 4: tools execute

    # STEP 1: user input enters the graph here
    graph.set_entry_point("agent")

    # STEP 3: conditional edge — tool needed OR final answer
    graph.add_conditional_edges(
        source="agent",
        path=should_use_tool,
        path_map={
            "call_tools": "call_tools",   # tool call → execute it
            "__end__": END,               # no tool call → return to user
        },
    )

    # STEP 5: after tool execution, always loop back to LLM for next reasoning step
    graph.add_edge("call_tools", "agent")

    # Compile with MemorySaver — full state persisted per thread_id
    return graph.compile(checkpointer=MemorySaver())