import streamlit as st
import uuid
import hashlib
from typing import Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

st.set_page_config(page_title="NEODesk", page_icon="🏦", layout="centered")

# ─────────────────────────────────────────────────────────────────────────────
# MOCK BANKING TOOLS
# Replace these with real API calls / DB queries in production
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_account_balance(account_type: str = "savings") -> str:
    """Get the current balance for a given account type (savings, current, fd)."""
    mock_data = {
        "savings":  "₹1,24,530.00",
        "current":  "₹3,82,900.00",
        "fd":        "₹5,00,000.00 (matures 12 Aug 2025)",
    }
    return mock_data.get(account_type.lower(), "Account type not found.")


@tool
def get_recent_transactions(limit: int = 5) -> str:
    """Fetch the most recent transactions. limit controls how many to return (max 10)."""
    txns = [
        "14 Jun · Zomato           −₹340",
        "13 Jun · Amazon           −₹2,199",
        "12 Jun · NEFT from Rahul  +₹10,000",
        "11 Jun · Electricity Bill −₹1,450",
        "10 Jun · ATM Withdrawal   −₹5,000",
        "09 Jun · Salary Credit    +₹85,000",
        "08 Jun · Netflix          −₹649",
        "07 Jun · Swiggy           −₹520",
    ]
    return "\n".join(txns[: min(limit, 10)])


@tool
def get_credit_score() -> str:
    """Fetch the customer's latest credit score and rating."""
    return "Credit Score: 762 / 900  |  Rating: GOOD  |  Last updated: 10 Jun 2025"


@tool
def check_loan_eligibility(loan_type: str = "personal", amount: int = 100000) -> str:
    """Check loan eligibility for a given loan type and amount (in INR)."""
    eligible = {
        "personal": amount <= 500000,
        "home":     amount <= 5000000,
        "car":      amount <= 1000000,
    }
    lt = loan_type.lower()
    if lt not in eligible:
        return f"Loan type '{loan_type}' not recognised. Try: personal, home, car."
    if eligible[lt]:
        return f"✅ Eligible for {lt} loan of ₹{amount:,}. Estimated EMI at 10.5% for 3 yrs: ₹{int(amount*0.0324):,}/month."
    return f"❌ Requested amount ₹{amount:,} exceeds limit for {lt} loan."


@tool
def get_card_status() -> str:
    """Get the status of all linked debit and credit cards."""
    return (
        "Debit Card  ····4521 · ACTIVE   · Limit: ₹50,000/day\n"
        "Credit Card ····8873 · ACTIVE   · Available: ₹45,230 / ₹80,000\n"
        "Virtual Card ···2290 · INACTIVE"
    )


TOOLS = [
    get_account_balance,
    get_recent_transactions,
    get_credit_score,
    check_loan_eligibility,
    get_card_status,
]

# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE CACHE  (saves LLM call for repeated identical prompts)
# Key = md5(normalised prompt).  Only caches short factual questions.
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

if "response_cache" not in st.session_state:
    st.session_state.response_cache: dict[str, str] = {}

def cache_get(prompt: str):
    return st.session_state.response_cache.get(_cache_key(prompt))

def cache_set(prompt: str, response: str):
    # Only cache short, stateless-looking questions (no "my", personalised words)
    stateful_keywords = {"transfer", "block", "unblock", "pay", "send", "change", "update"}
    if any(w in prompt.lower() for w in stateful_keywords):
        return  # don't cache action-oriented prompts
    if len(st.session_state.response_cache) > 200:  # cap size
        st.session_state.response_cache.clear()
    st.session_state.response_cache[_cache_key(prompt)] = response

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are NEODesk, a professional AI banking assistant.

You have access to real tools to fetch account data. Always use them when a customer
asks for balance, transactions, credit score, loan eligibility, or card status —
don't guess or make up numbers.

Rules:
- Never ask for full card numbers, PINs, or passwords.
- For irreversible actions (fund transfer, card block), confirm intent first.
- If a tool isn't available for a request, say so and suggest calling the helpline.
- Keep answers concise and structured. Use bullet points for lists."""

# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH AGENT
# ─────────────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


@st.cache_resource
def get_graph():
    llm = ChatOllama(model="llama3.1:8b", temperature=0.3, streaming=True)
    llm_with_tools = llm.bind_tools(TOOLS)

    def maybe_trim(state: State) -> State:
        """Keep only the last 16 messages to avoid context overflow."""
        trimmed = trim_messages(
            state["messages"],
            max_tokens=16,          # message count, not tokens (strategy="last")
            token_counter=len,
            strategy="last",
            include_system=False,
            allow_partial=False,
        )
        return {"messages": trimmed}

    def chatbot(state: State) -> State:
        system = SystemMessage(content=SYSTEM_PROMPT)
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools=TOOLS)

    builder = StateGraph(State)
    builder.add_node("trim",    maybe_trim)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools",   tool_node)

    builder.add_edge(START,      "trim")
    builder.add_edge("trim",     "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)  # → tools or END
    builder.add_edge("tools",    "chatbot")                    # loop back after tool call

    return builder.compile(checkpointer=MemorySaver())


graph = get_graph()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("💬 NEODesk")
    st.caption("Thread ID")
    st.code(st.session_state.thread_id[:24] + "...", language=None)

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Messages this session: {len(st.session_state.messages)}")
    cache_hits = st.session_state.get("cache_hit_count", 0)
    st.caption(f"Cache hits: {cache_hits}")

    st.divider()
    st.caption("**Available tools**")
    for t in TOOLS:
        st.caption(f"• `{t.name}`")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏦 NEODesk")
st.caption("AI Banking Assistant · llama3.1:8b · LangGraph · MemorySaver")
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me about your account, cards, loans..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # ── Check cache first ──────────────────────────────────────────────────
    cached = cache_get(prompt)
    if cached:
        st.session_state["cache_hit_count"] = st.session_state.get("cache_hit_count", 0) + 1
        with st.chat_message("assistant"):
            st.write(cached)
            st.caption("⚡ cached response")
        st.session_state.messages.append({"role": "assistant", "content": cached})
        st.rerun()

    # ── Stream from LangGraph ──────────────────────────────────────────────
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk, _ in graph.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="messages",
            ):
                if hasattr(chunk, "content") and chunk.content:
                    # Skip tool call chunks (they're not readable text)
                    if getattr(chunk, "type", None) == "tool":
                        continue
                    full_response += chunk.content
                    placeholder.write(full_response + "▍")

            placeholder.write(full_response)

        except Exception as e:
            full_response = f"❌ Error: {e}\n\nMake sure Ollama is running:\n`ollama run llama3.1:8b`"
            placeholder.error(full_response)

    if full_response:
        cache_set(prompt, full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})