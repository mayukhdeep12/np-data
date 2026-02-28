import streamlit as st
import uuid
import asyncio
import queue
import threading
from pathlib import Path

from langchain_core.messages import HumanMessage
from mcp_manager import MCPManager
from agent import build_agent, BUILTIN_TOOLS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NEODesk", page_icon="🏦", layout="centered")

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — must be initialized FIRST before any other code runs
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""
if "mcp_manager" not in st.session_state:
    st.session_state.mcp_manager = None
if "mcp_connected" not in st.session_state:
    st.session_state.mcp_connected = False
if "mcp_tool_names" not in st.session_state:
    st.session_state.mcp_tool_names = set()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-1"
if "_tmp_loop" not in st.session_state:
    st.session_state._tmp_loop = None

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_mcp_tool(name: str) -> bool:
    return name in st.session_state.mcp_tool_names


def tool_badge(name: str) -> str:
    if is_mcp_tool(name):
        return f"🔌 MCP  ·  `{name}`"
    return f"⚙️ Built-in  ·  `{name}`"


def rebuild_agent():
    """Recompile the LangGraph agent with current tools. Safe to call anytime."""
    if not st.session_state.groq_key:
        return
    mcp_tools = st.session_state.mcp_manager.tools if st.session_state.mcp_connected else []
    try:
        st.session_state.agent = build_agent(
            groq_api_key=st.session_state.groq_key,
            extra_tools=mcp_tools,
        )
    except Exception as e:
        st.error(f"Agent build failed: {e}")


def get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Return the event loop to use for running the agent.
    Prefers the MCP manager's loop so MCP async tools stay in scope.
    Falls back to a standalone background loop when MCP is not connected.
    """
    if st.session_state.mcp_manager is not None:
        return st.session_state.mcp_manager.loop

    if st.session_state._tmp_loop is None:
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        st.session_state._tmp_loop = loop

    return st.session_state._tmp_loop


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Settings")

    # ── 1. Groq API Key ────────────────────────────────────────────────────────
    st.subheader("1. Groq API Key")

    if not st.session_state.groq_key:
        st.info("Get a free key at [console.groq.com](https://console.groq.com)")

    groq_input = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_key,
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
    )

    # Key changed → rebuild agent
    if groq_input != st.session_state.groq_key:
        st.session_state.groq_key = groq_input
        st.session_state.agent = None
        if groq_input:
            rebuild_agent()

    # First load with key already present → build agent
    if st.session_state.groq_key and st.session_state.agent is None:
        rebuild_agent()

    st.markdown("---")

    # ── 2. MCP Banking Server ──────────────────────────────────────────────────
    st.subheader("2. MCP Banking Server")

    default_mcp_path = str(
        Path(__file__).parent.parent / "neodesk_mcp" / "banking_mcp_server.py"
    )
    mcp_path = st.text_input(
        "Path to banking_mcp_server.py",
        value=default_mcp_path,
        placeholder="/absolute/path/to/banking_mcp_server.py",
        help="Absolute path to your local banking_mcp_server.py",
    )

    col1, col2 = st.columns(2)

    with col1:
        connect_clicked = st.button(
            "Connect",
            use_container_width=True,
            type="primary",
            disabled=(not st.session_state.groq_key),
        )
    with col2:
        disconnect_clicked = st.button(
            "Disconnect",
            use_container_width=True,
            disabled=(not st.session_state.mcp_connected),
        )

    # Handle Connect
    if connect_clicked:
        if not mcp_path or not Path(mcp_path).exists():
            st.error(f"File not found:\n`{mcp_path}`")
        else:
            with st.spinner("Connecting to MCP server…"):
                try:
                    if st.session_state.mcp_manager is None:
                        st.session_state.mcp_manager = MCPManager()
                    tools = st.session_state.mcp_manager.connect(mcp_path)
                    st.session_state.mcp_connected  = True
                    st.session_state.mcp_tool_names = {t.name for t in tools}
                    rebuild_agent()
                    st.success(f"Connected — {len(tools)} MCP tools loaded")
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    # Handle Disconnect
    if disconnect_clicked:
        if st.session_state.mcp_manager:
            st.session_state.mcp_manager.disconnect()
        st.session_state.mcp_connected  = False
        st.session_state.mcp_tool_names = set()
        rebuild_agent()
        st.rerun()

    # MCP status indicator
    if st.session_state.mcp_connected:
        st.success(f"🔌 Connected — {len(st.session_state.mcp_tool_names)} tools active")
        with st.expander("MCP Tools"):
            for name in sorted(st.session_state.mcp_tool_names):
                st.caption(f"• {name}")
    else:
        st.warning("🔌 Not connected")
        st.caption("Connect MCP to enable live account data.")

    st.markdown("---")

    # ── 3. Built-in tools ──────────────────────────────────────────────────────
    with st.expander("⚙️ Built-in Tools (always active)"):
        for t in BUILTIN_TOOLS:
            st.caption(f"• {t.name}")

    st.markdown("---")

    # ── 4. Session / Thread ────────────────────────────────────────────────────
    st.subheader("3. Session")

    thread_input = st.text_input(
        "Thread ID",
        value=st.session_state.thread_id,
        help="Each unique Thread ID has its own memory.",
    )
    if thread_input != st.session_state.thread_id:
        st.session_state.thread_id = thread_input
        st.session_state.messages  = []
        st.rerun()

    col3, col4 = st.columns(2)
    with col3:
        if st.button("New Thread", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.session_state.messages  = []
            st.rerun()
    with col4:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption(f"Model : Llama 3.3 70B (Groq)")
    st.caption(f"Thread: {st.session_state.thread_id}")
    st.caption(f"Msgs  : {len(st.session_state.messages)}")


# ══════════════════════════════════════════════════════════════════════════════
# GATE — require API key before showing chat
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏦 NEODesk Banking Assistant")

if not st.session_state.groq_key:
    st.info("👈 Enter your **Groq API key** in the sidebar to get started.\n\nGet a free key at [console.groq.com](https://console.groq.com)")
    st.stop()

if not st.session_state.mcp_connected:
    st.info(
        "💡 **Tip:** Connect the MCP Banking Server in the sidebar to enable live "
        "account data (balances, cards, fraud detection, transfers, etc.).\n\n"
        "Built-in tools (loan rates, FAQs) work without MCP."
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            badges = "  |  ".join(tool_badge(n) for n in msg["tool_calls"])
            st.caption(f"Tools used → {badges}")
        st.write(msg["content"])


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT + STREAMING RESPONSE
# ══════════════════════════════════════════════════════════════════════════════
if prompt := st.chat_input("Ask about your accounts, cards, transactions…"):

    if not st.session_state.agent:
        st.error("Agent not ready. Check your Groq API key in the sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        tool_calls_this_turn = []
        full_response        = ""

        tool_placeholder = st.empty()
        resp_placeholder = st.empty()

        result_q = queue.Queue()
        config   = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Capture as plain local variables BEFORE the async def.
        # st.session_state is NOT accessible inside async functions
        # running in a background thread — this is the root cause of the error.
        _agent  = st.session_state.agent
        _prompt = prompt

        async def run_stream():
            try:
                async for chunk, _ in _agent.astream(
                    {"messages": [HumanMessage(content=_prompt)]},
                    config=config,
                    stream_mode="messages",
                ):
                    result_q.put(("chunk", chunk))
                result_q.put(("done", None))
            except Exception as exc:
                result_q.put(("error", str(exc)))

        asyncio.run_coroutine_threadsafe(run_stream(), get_event_loop())

        while True:
            try:
                kind, chunk = result_q.get(timeout=60)
            except queue.Empty:
                resp_placeholder.error("Request timed out. Please try again.")
                break

            if kind == "done":
                break

            if kind == "error":
                err = chunk
                if "failed_generation" in err or "Failed to call a function" in err:
                    full_response = "I had trouble with that request. Could you rephrase it?"
                    resp_placeholder.write(full_response)
                else:
                    full_response = f"Error: {err}"
                    resp_placeholder.error(full_response)
                break

            # LLM decided to call a tool
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tc in chunk.tool_calls:
                    name = tc.get("name", "")
                    if name and name not in tool_calls_this_turn:
                        tool_calls_this_turn.append(name)
                badges = "  |  ".join(tool_badge(n) for n in tool_calls_this_turn)
                tool_placeholder.caption(f"🛠 Tools used → {badges}")

            # Stream text tokens
            if hasattr(chunk, "content") and chunk.content:
                if not hasattr(chunk, "tool_call_id"):
                    full_response += chunk.content
                    resp_placeholder.write(full_response + "▌")

        if full_response:
            resp_placeholder.write(full_response)

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    full_response,
        "tool_calls": tool_calls_this_turn,
    })