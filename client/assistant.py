import asyncio
import concurrent.futures
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Safely check Streamlit secrets if running in Streamlit Cloud
def _get_config(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

API_KEY = _get_config("GROQ_API_KEY")
BASE_URL = _get_config("LLM_BASE_URL", "https://api.groq.com/openai/v1")

# LLM_MODELS: comma-separated, tried in order when a model is rate-limited.
_models = _get_config("LLM_MODELS") or _get_config("LLM_MODEL") or ""
MODELS = [m.strip() for m in _models.split(",") if m.strip()]

if not API_KEY or not MODELS:
    raise RuntimeError(
        "Missing configuration: Set GROQ_API_KEY and LLM_MODELS in .env (local) "
        "or under App Settings > Secrets in Streamlit Cloud."
    )

MODEL = MODELS[0]  # kept for any script that still imports MODEL

# The SDK retries short per-minute limits by itself (it honours retry-after).
# A used-up daily quota isn't worth waiting for: after two quick retries
# (about 1.5 s) the error reaches chat_with_fallback, which tries the next model.
llm = OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=2)

SYSTEM_PROMPT = (
    "You are MLOps Copilot. Use the tools to answer questions about ML "
    "experiments, registered models, and deployment status. Never guess "
    "numbers: call a tool. When you rank runs with compare_runs, say which "
    "metric you ranked by. Only say results were ranked or compared if you "
    "called compare_runs. For retraining, call trigger_retrain with the "
    "parameters only; the app asks the user for approval itself. Never ask "
    "for or invent tokens. After a retrain, just report the run ID, the "
    "version and the metrics. If a tool returns an error, explain it "
    "plainly. You have no tool to promote a model, change the champion "
    "alias, or deploy. Never offer those actions; say a human must do them. "
    "If two values are tied or nearly equal, say so instead of calling one "
    "better."
)

MAX_STEPS = 4          # LLM calls allowed per question
TOOL_OUT_CHARS = 1500  # tool output sent to the model; the UI keeps the full text
MAX_HISTORY = 12       # messages after the system prompt kept between questions


def to_llm_tools(mcp_tools):
    """Convert MCP tool definitions to the OpenAI function-calling format."""
    tools = []
    for t in mcp_tools:
        schema = json.loads(json.dumps(t.inputSchema))  # deep copy
        if t.name == "trigger_retrain":
            # The model must never see or use the confirmation token.
            schema.get("properties", {}).pop("confirm_token", None)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": schema,
                },
            }
        )
    return tools


def chat_with_fallback(messages, tools):
    """Try each model in order; move to the next one only on a rate limit."""
    last_err = None
    for model in MODELS:
        try:
            return llm.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto"
            )
        except RateLimitError as e:
            print(f"  [model] {model} rate-limited, trying next...")
            last_err = e
    raise last_err  # every model in LLM_MODELS is rate-limited


def trim_history(messages):
    """Keep the system prompt and the last MAX_HISTORY messages.

    The kept part must not start with a tool result whose tool call was cut
    off, because the API rejects such a history.
    """
    system, rest = messages[:1], messages[1:]
    if len(rest) > MAX_HISTORY:
        rest = rest[-MAX_HISTORY:]
        while rest and rest[0]["role"] == "tool":
            rest = rest[1:]
    messages[:] = system + rest


def friendly_error(e):
    """One sentence the user can act on, instead of a raw API error."""
    if isinstance(e, RateLimitError):
        text = str(e)
        m = re.search(r"try again in ([\dhms.]+)", text)
        wait = m.group(1).rstrip(".") if m else "a few minutes"
        if "per day" in text:
            return (
                f"The free daily quota for the AI model is used up. Try again in "
                f"about {wait}, or add another model to LLM_MODELS in .env."
            )
        return f"The AI model is busy (rate limit). Try again in about {wait}."
    if isinstance(e, AuthenticationError):
        return "The AI API rejected the key. Check GROQ_API_KEY in .env."
    if isinstance(e, APIConnectionError):
        return "Could not reach the AI API. Check your internet connection and LLM_BASE_URL."
    if isinstance(e, (TimeoutError, concurrent.futures.TimeoutError, asyncio.TimeoutError)):
        return "A tool took more than 3 minutes and was stopped. Try again, or check logs/server_stderr.log."
    return f"Request failed: {str(e) or type(e).__name__}"


async def call_tool(session, name, args):
    result = await session.call_tool(name, args)
    text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
    if result.isError:
        text = f"TOOL ERROR: {text}"
    return text


# ---------- Terminal client (the Streamlit app has its own loop) ----------
async def retrain_with_approval(session, args):
    """The human approves here, in the terminal. The model never sees the token."""
    args = {k: v for k, v in args.items() if k != "confirm_token"}
    preview_text = await call_tool(session, "trigger_retrain", args)
    try:
        preview = json.loads(preview_text)
    except json.JSONDecodeError:
        return preview_text  # error or unexpected output: show it to the model
    if preview.get("status") != "CONFIRMATION_REQUIRED":
        return preview_text  # e.g. BLOCKED in read-only mode

    print(f"\n--- APPROVAL NEEDED ---\n{preview['preview']}")
    answer = input("Approve this action? (y/n): ").strip().lower()
    if answer != "y":
        return json.dumps({"status": "CANCELLED_BY_USER"})

    print("Running... this can take up to a minute.")
    return await call_tool(
        session, "trigger_retrain", {**args, "confirm_token": preview["confirm_token"]}
    )


async def run_turn(session, messages, tools):
    trim_history(messages)
    for _ in range(MAX_STEPS):
        resp = chat_with_fallback(messages, tools)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "(no answer)"

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

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  [tool] {name}({args})")

            if name == "trigger_retrain":
                output = await retrain_with_approval(session, args)
            else:
                output = await call_tool(session, name, args)

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": output[:TOOL_OUT_CHARS]}
            )

    return "Stopped: too many tool steps for one question."


async def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "server" / "mlflow_server.py")]
    )
    with open(ROOT / "logs" / "server_stderr.log", "a") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = to_llm_tools((await session.list_tools()).tools)
                print(f"Connected. {len(tools)} tools loaded. Models: {', '.join(MODELS)}")
                print("Type 'quit' to exit.\n")

                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                while True:
                    question = input("You: ").strip()
                    if question.lower() in {"quit", "exit"}:
                        break
                    if not question:
                        continue
                    mark = len(messages)
                    messages.append({"role": "user", "content": question})
                    try:
                        print("\nCopilot:", await run_turn(session, messages, tools), "\n")
                    except Exception as e:
                        del messages[mark:]  # keep the history consistent
                        print("\nCopilot:", friendly_error(e), "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")