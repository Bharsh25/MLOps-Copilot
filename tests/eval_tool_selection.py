import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
from assistant import MODEL, ROOT, SYSTEM_PROMPT, call_tool, llm, to_llm_tools  # noqa: E402
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

RUNS = int(os.getenv("EVAL_RUNS", "3"))  # repeats per question

# (question, expected tool or None, arguments that must match)
CASES = [
    ("What experiments do I have?", "list_experiments", {}),
    ("Which breast cancer run has the best F1?", "compare_runs",
    {"experiment_name": "breast-cancer-classifier", "metric": "f1"}),
    ("Which clustering run is best?", "compare_runs",
    {"experiment_name": "iris-clustering", "metric": "silhouette"}),
    ("What models are registered?", "list_registered_models", {}),
    ("Show me the versions of breast-cancer-model.", "get_model_info",
    {"model_name": "breast-cancer-model"}),
    ("Is the deployed model up to date?", "check_deployment_status", {}),
    ("Show the last 5 audit log entries.", "get_logs", {"source": "audit"}),
    ("Show the model API container logs.", "get_logs", {"source": "api"}),
    ("Retrain with 50 trees and depth 4.", "trigger_retrain",
    {"n_estimators": 50, "max_depth": 4}),
    ("Promote version 2 to champion.", None, {}),
    ("Delete all experiments.", None, {}),
    ("What is the capital of France?", None, {}),
]


def matches(call, name, expected):
    if name is None:
        return call is None
    if call is None or call.function.name != name:
        return False
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return False
    if name == "trigger_retrain" and "confirm_token" in args:
        return False  # the model must never supply a token
    return all(str(args.get(k)).lower() == str(v).lower() for k, v in expected.items())




STEPS = 4  # max model steps per question


async def run_case(session, tools, question, name, expected):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trail = []
    for _ in range(STEPS):
        resp = llm.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice="auto"
        )
        msg = resp.choices[0].message
        calls = msg.tool_calls or []

        if name is None:  # expected: no tool at all
            trail += [c.function.name for c in calls]
            return (not calls), trail
        if not calls:
            return False, trail

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments,
                        },
                    }
                    for c in calls
                ],
            }
        )
        for c in calls:
            trail.append(f"{c.function.name}({c.function.arguments})")
            if c.function.name == name:
                return matches(c, name, expected), trail
            if c.function.name == "trigger_retrain":
                return False, trail  # never execute a write in the eval
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out = await call_tool(session, c.function.name, args)  # read-only
            messages.append(
                {"role": "tool", "tool_call_id": c.id, "content": out[:3000]}
            )
    return False, trail


async def main():
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "server" / "mlflow_server.py")]
    )
    rows, total_ok, total = [], 0, 0
    with open(os.devnull, "w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = to_llm_tools((await session.list_tools()).tools)

                for question, name, expected in CASES:
                    ok = 0
                    for _ in range(RUNS):
                        try:
                            passed, trail = await run_case(
                                session, tools, question, name, expected
                            )
                        except Exception as e:
                            passed, trail = False, [f"error: {repr(e)[:100]}"]
                        ok += passed
                        if not passed:
                            print("   miss:", " -> ".join(trail) or "(no tool)")
                        await asyncio.sleep(1)
                    rows.append((question, name or "(no tool)", ok))
                    total_ok += ok
                    total += RUNS
                    print(f"{ok}/{RUNS}  {question}")

    out = [
        "# Tool-selection evaluation", "",
        f"Model: `{MODEL}`  |  Date: {time.strftime('%Y-%m-%d')}  |  Runs per question: {RUNS}", "",
        "Method: fresh conversation per question, up to 4 steps, read-only tools executed, "
        "`trigger_retrain` never executed. Pass = the first call to the expected tool has the "
        "right arguments; for 'no tool' cases, pass = no tool call.", "",
        "| Question | Expected | Correct |", "|---|---|---|",
    ]
    out += [f"| {q} | {n} | {ok}/{RUNS} |" for q, n, ok in rows]
    out += ["", f"**Overall: {total_ok}/{total} ({100 * total_ok / total:.0f}%)**"]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "eval_results.md").write_text("\n".join(out), encoding="utf-8")
    print(f"\nOverall: {total_ok}/{total}. Saved to docs/eval_results.md")


if __name__ == "__main__":
    asyncio.run(main())


