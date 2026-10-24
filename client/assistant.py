import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL")
if not API_KEY or not MODEL:
    sys.exit("Set GROQ_API_KEY and LLM_MODEL in .env")

llm = OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=5)

SYSTEM_PROMPT = (
    "You are MLOps Copilot. Use the tools to answer questions about ML "
    "experiments, registered models, and deployment status. Never guess "
    "numbers: call a tool. Always say which metric you ranked by. For "
    "retraining, call trigger_retrain with the parameters only; the app "
    "asks the user for approval itself. Never ask for or invent tokens. "
    "If a tool returns an error, explain it plainly."
    "You have no tool to promote a model, change the champion alias, or "
    "deploy. Never offer those actions; say a human must do them. If two "
    "values are tied or nearly equal, say so instead of calling one better."
)

MAX_STEPS = 6


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


async def call_tool(session, name, args):
    result = await session.call_tool(name, args)
    text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
    if result.isError:
        text = f"TOOL ERROR: {text}"
    return text


async def retrain_with_approval(session, args):
    """The human approves here, in the app. The model never sees the token."""
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
    for _ in range(MAX_STEPS):
        resp = llm.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice="auto"
        )
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
                {"role": "tool", "tool_call_id": tc.id, "content": output[:6000]}
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
                print(f"Connected. {len(tools)} tools loaded. Type 'quit' to exit.\n")

                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                while True:
                    question = input("You: ").strip()
                    if question.lower() in {"quit", "exit"}:
                        break
                    if not question:
                        continue
                    messages.append({"role": "user", "content": question})
                    print("\nCopilot:", await run_turn(session, messages, tools), "\n")


if __name__ == "__main__":
    asyncio.run(main())