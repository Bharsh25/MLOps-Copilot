import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assistant import (  # noqa: E402
    MAX_STEPS, ROOT, SYSTEM_PROMPT, TOOL_OUT_CHARS, call_tool,
    chat_with_fallback, friendly_error, to_llm_tools, trim_history,
)

READ_ONLY = os.getenv("MLOPS_READ_ONLY", "true").lower() != "false"

st.set_page_config(page_title="MLOps Copilot", page_icon="🛠️")

WELCOME = (
    "👋 **Welcome to MLOps Copilot!**\n\n"
    "I'm connected to your MLflow tracking server and model registry. Here is what I can do:\n"
    "- **Experiment tracking:** list experiments and compare run metrics.\n"
    "- **Registry management:** check model versions, champion tags and status.\n"
    "- **CI/CD & retraining:** inspect deployment status or trigger a retrain (with your approval).\n\n"
    "*Click a quick prompt above, or type a question below to start.*"
)

PRESETS = {
    "deploy": "Check the deployment status of breast-cancer-model and see if it matches champion version.",
    "compare": "Compare the latest runs and show me which one had the best performance.",
    "models": "List all registered models and their active versions.",
}


# ---------- MCP connection that survives Streamlit reruns ----------
class Bridge:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.session = None
        self.tools = []
        self.error = None
        ready = threading.Event()
        asyncio.run_coroutine_threadsafe(self._serve(ready), self.loop)
        if not ready.wait(90):
            raise RuntimeError("MCP server did not start within 90 seconds.")
        if self.error:
            raise RuntimeError(self.error)

    async def _serve(self, ready):
        try:
            (ROOT / "logs").mkdir(exist_ok=True)
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(ROOT / "server" / "mlflow_server.py")],
            )
            with open(ROOT / "logs" / "server_stderr.log", "a") as errlog:
                async with stdio_client(params, errlog=errlog) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self.tools = to_llm_tools((await session.list_tools()).tools)
                        self.session = session
                        ready.set()
                        await asyncio.Event().wait()  # keep the server alive
        except Exception as e:
            self.error = repr(e)
            ready.set()

    def call(self, name, args):
        fut = asyncio.run_coroutine_threadsafe(
            call_tool(self.session, name, args), self.loop
        )
        return fut.result(timeout=180)


@st.cache_resource
def get_bridge():
    return Bridge()


# ---------- Agent loop that can pause for human approval ----------
def advance(bridge, messages, trace):
    """Run the model until it answers, or until a retrain needs approval."""
    for _ in range(MAX_STEPS):
        resp = chat_with_fallback(messages, bridge.tools)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return "answer", msg.content or "(no answer)"

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        pending = None
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "trigger_retrain":
                args.pop("confirm_token", None)  # the model never supplies a token
                if pending is not None:
                    out = json.dumps(
                        {"status": "SKIPPED", "reason": "Only one retrain at a time."}
                    )
                else:
                    out = bridge.call(name, args)
                    try:
                        preview = json.loads(out)
                    except json.JSONDecodeError:
                        preview = {}
                    if preview.get("status") == "CONFIRMATION_REQUIRED":
                        # Wait for the human; the tool message is added afterwards.
                        pending = {
                            "tool_call_id": tc.id,
                            "args": args,
                            "token": preview["confirm_token"],
                            "preview": preview["preview"],
                        }
                        continue
            else:
                out = bridge.call(name, args)

            trace.append({"name": name, "args": args, "output": out})  # full text for the UI
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": out[:TOOL_OUT_CHARS]}
            )

        if pending:
            return "approval", pending

    return "answer", "Stopped: too many tool steps for one question."


def finish(kind, value):
    ss = st.session_state
    if kind == "answer":
        ss.chat.append({"role": "assistant", "content": value, "tools": ss.trace})
        ss.trace = []
    else:
        ss.pending = value


# ---------- State ----------
ss = st.session_state
ss.setdefault("messages", [{"role": "system", "content": SYSTEM_PROMPT}])  # sent to the LLM
ss.setdefault("chat", [])                                                  # shown in the UI
ss.setdefault("pending", None)
ss.setdefault("trace", [])
ss.setdefault("error", None)

# Welcome text lives only in the UI history, so it is never sent to the LLM.
if not ss.chat:
    ss.chat.append({"role": "assistant", "content": WELCOME})

try:
    bridge = get_bridge()
except Exception as e:
    st.error(f"Could not start the MCP server: {e}")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Status")
    st.write("Write actions:", "🔒 read-only" if READ_ONLY else "✍️ enabled (needs approval)")
    refresh = st.button("Refresh status")
    if refresh or "status" not in ss:
        try:
            ss.status = json.loads(bridge.call("check_deployment_status", {}))
        except Exception as e:
            ss.status = {"status": "UNKNOWN", "error": str(e)}
    s = ss.status
    st.write("Model API:", "🟢 UP" if s.get("status") == "UP" else f"🔴 {s.get('status')}")
    st.write("Deployed version:", s.get("deployed_version", "-"))
    st.write("Champion version:", s.get("champion_version", "-"))
    if s.get("status") == "UP":
        st.write("Up to date:", "✅" if s.get("up_to_date") else "⚠️ no")
    if st.button("Clear chat"):
        for k in ("messages", "chat", "pending", "trace", "error"):
            ss.pop(k, None)
        st.rerun()

# ---------- Header, errors, quick prompts ----------
st.title("MLOps Copilot")
st.caption("AI-powered assistant for MLflow experiment tracking, model registry, and deployment checks.")

if ss.error:
    st.error(ss.error)
    ss.error = None

busy = bool(ss.pending)  # no new questions while an approval is waiting

st.markdown("##### ⚡ Quick Prompts")
c1, c2, c3 = st.columns(3)
preset_query = None
if c1.button("🔍 Check Deployment", disabled=busy):
    preset_query = PRESETS["deploy"]
if c2.button("📊 Compare Runs", disabled=busy):
    preset_query = PRESETS["compare"]
if c3.button("📋 List All Models", disabled=busy):
    preset_query = PRESETS["models"]

# ---------- Chat history ----------
for item in ss.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("tools"):
            with st.expander(f"Tools used ({len(item['tools'])})"):
                for t in item["tools"]:
                    st.code(f"{t['name']}({json.dumps(t['args'])})", language="python")
                    st.code(t["output"][:1500], language="json")

# ---------- Approval box ----------
if ss.pending:
    p = ss.pending
    st.warning(f"**Approval needed**\n\n{p['preview']}")
    a1, a2 = st.columns(2)
    approve = a1.button("✅ Approve", type="primary")
    cancel = a2.button("❌ Cancel")
    if approve or cancel:
        try:
            if approve:
                with st.spinner("Retraining... this can take up to 2 minutes."):
                    out = bridge.call(
                        "trigger_retrain", {**p["args"], "confirm_token": p["token"]}
                    )
            else:
                out = json.dumps({"status": "CANCELLED_BY_USER"})
        except Exception as e:
            # Still answer the tool call, or the LLM history becomes invalid.
            out = json.dumps({"status": "FAILED", "error": str(e)})

        ss.trace.append({"name": "trigger_retrain", "args": p["args"], "output": out})
        ss.messages.append(
            {"role": "tool", "tool_call_id": p["tool_call_id"], "content": out[:TOOL_OUT_CHARS]}
        )
        ss.pending = None
        try:
            with st.spinner("Thinking..."):
                finish(*advance(bridge, ss.messages, ss.trace))
        except Exception as e:
            ss.error = friendly_error(e)
        st.rerun()

# ---------- Input (typed question or quick prompt) ----------
prompt = st.chat_input(
    "Ask about experiments, models, or deployment...", disabled=busy
) or preset_query

if prompt and not ss.pending:
    trim_history(ss.messages)  # before the marks, so the rollback stays correct
    mark_msgs, mark_chat = len(ss.messages), len(ss.chat)
    ss.chat.append({"role": "user", "content": prompt})
    ss.messages.append({"role": "user", "content": prompt})
    ss.trace = []

    with st.chat_message("user"):  # show the question now, not after the rerun
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                finish(*advance(bridge, ss.messages, ss.trace))
    except Exception as e:
        del ss.messages[mark_msgs:]  # keep both histories consistent
        del ss.chat[mark_chat:]
        ss.trace = []
        ss.error = friendly_error(e)
    st.rerun()