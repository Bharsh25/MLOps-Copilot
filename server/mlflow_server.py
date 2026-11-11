import json
import logging
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import mlflow
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mlflow.tracking import MlflowClient

# 1. Resolve root and load environment first
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 2. Database URI (POSIX slashes work on Windows and Linux)
DB_PATH = ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}?timeout=30"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# 3. Logging and config
# IMPORTANT: stdio servers must never print() to stdout (reserved for the MCP protocol).
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("mlops-copilot")

# 127.0.0.1 instead of "localhost": in some Linux containers "localhost" resolves to
# IPv6 (::1) and fails with "[Errno 99] Cannot assign requested address".
# Set MODEL_API_URL to an empty value (or "none") when no model API exists,
# for example on a hosted demo.
API_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8000").strip()
if API_URL.lower() in {"none", "off", "disabled", "false"}:
    API_URL = ""

READ_ONLY = os.getenv("MLOPS_READ_ONLY", "true").lower() != "false"
AUDIT_LOG = ROOT / "logs" / "audit.jsonl"
TOKEN_TTL = 300  # seconds a confirmation token stays valid
PENDING: dict[str, dict] = {}

# Names that reach MLflow filters or ORDER BY clauses must be plain identifiers.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-/]{1,100}$")


def _check_name(value: str, label: str) -> str:
    if not _SAFE_NAME.match(value or ""):
        raise ValueError(
            f"{label} may only contain letters, numbers and _ . - / (max 100 characters)."
        )
    return value


def audit(event: str, **details) -> bool:
    """Append one JSON line to the audit log. Returns False if it could not be written."""
    try:
        AUDIT_LOG.parent.mkdir(exist_ok=True)
        entry = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except OSError as e:
        log.error("audit log could not be written: %s", e)
        return False


mcp = FastMCP("mlops-copilot")


@mcp.tool()
def list_experiments() -> list[dict]:
    """List all ML experiments with their experiment ID and number of runs.
    Use this first to discover which experiments exist before comparing runs."""
    log.info("tool called: list_experiments")
    results = []
    for exp in client.search_experiments():
        runs = client.search_runs([exp.experiment_id], max_results=1000)
        results.append(
            {
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "run_count": len(runs),
            }
        )
    return results


@mcp.tool()
def compare_runs(
    experiment_name: str,
    metric: str,
    top_n: int = 5,
    higher_is_better: bool = True,
) -> list[dict]:
    """Rank finished runs of one experiment by a metric.

    Args:
        experiment_name: Exact experiment name, e.g. 'breast-cancer-classifier'.
        metric: Metric to rank by, e.g. 'f1', 'accuracy', or 'silhouette'.
        top_n: How many top runs to return, 1 to 50 (default 5).
        higher_is_better: True for accuracy, f1 and silhouette. False for error
            metrics such as rmse and for Davies-Bouldin, where lower is better.
    """
    log.info("tool called: compare_runs(%s, %s)", experiment_name, metric)
    _check_name(metric, "metric")
    if not 1 <= top_n <= 50:
        raise ValueError("top_n must be between 1 and 50.")

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. Call list_experiments first."
        )

    direction = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} {direction}"],
        max_results=top_n,
    )
    return [
        {
            "run_name": r.info.run_name,
            "run_id": r.info.run_id,
            "params": r.data.params,
            "metrics": r.data.metrics,
        }
        for r in runs
    ]


@mcp.tool()
def list_registered_models() -> list[dict]:
    """List all registered models with their latest version and aliases.
    Use this to discover which models are available before asking for details."""
    log.info("tool called: list_registered_models")
    return [
        {
            "name": m.name,
            "aliases": dict(m.aliases or {}),
            "latest_versions": [v.version for v in (m.latest_versions or [])],
        }
        for m in client.search_registered_models()
    ]


@mcp.tool()
def get_model_info(model_name: str) -> dict:
    """Get all versions of one registered model, including the source run's
    metrics and which version has the 'champion' alias (the one to deploy).

    Args:
        model_name: Exact registered model name, e.g. 'breast-cancer-model'.
    """
    log.info("tool called: get_model_info(%s)", model_name)
    _check_name(model_name, "model_name")
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(
            f"Model '{model_name}' not found. Call list_registered_models first."
        )

    # Aliases live on the registered model: {alias: version}. Invert to {version: [aliases]}.
    registered = client.get_registered_model(model_name)
    aliases_by_version = {}
    for alias, ver in (registered.aliases or {}).items():
        aliases_by_version.setdefault(str(ver), []).append(alias)

    result = []
    for v in versions:
        run = client.get_run(v.run_id)
        result.append(
            {
                "version": v.version,
                "aliases": aliases_by_version.get(str(v.version), []),
                "run_id": v.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }
        )
    return {"model_name": model_name, "versions": result}


@mcp.tool()
def check_deployment_status(model_name: str = "breast-cancer-model") -> dict:
    """Check whether the deployed model API is reachable and whether it serves
    the current 'champion' version from the registry. Report the result exactly
    as returned. If the status is DOWN, the cause is not known to this tool.

    Args:
        model_name: Registered model name (default 'breast-cancer-model').
    """
    log.info("tool called: check_deployment_status(%s)", model_name)
    _check_name(model_name, "model_name")
    try:
        registered = client.get_registered_model(model_name)
    except Exception:
        raise ValueError(
            f"Model '{model_name}' not found. Call list_registered_models first."
        )
    champion = (registered.aliases or {}).get("champion")
    champion_version = str(champion) if champion is not None else None

    if not API_URL:
        return {
            "status": "NOT_CONFIGURED",
            "detail": "No model API is set up for this deployment (MODEL_API_URL is empty).",
            "champion_version": champion_version,
        }

    try:
        resp = httpx.get(f"{API_URL}/health", timeout=10)
        resp.raise_for_status()
        health = resp.json()
    except httpx.HTTPStatusError as e:
        return {
            "status": "DOWN",
            "detail": f"The API at {API_URL} answered with HTTP {e.response.status_code}.",
            "champion_version": champion_version,
        }
    except Exception as e:
        return {
            "status": "DOWN",
            "detail": (
                f"Could not connect to the API at {API_URL} "
                f"({type(e).__name__}: {str(e)[:150]})."
            ),
            "champion_version": champion_version,
        }

    deployed = str(health.get("model_version"))
    return {
        "status": "UP",
        "serving_model": health.get("model_name"),
        "deployed_version": deployed,
        "champion_version": champion_version,
        "up_to_date": deployed == champion_version,
    }


@mcp.tool()
def trigger_retrain(
    n_estimators: int = 100,
    max_depth: int | None = None,
    confirm_token: str | None = None,
) -> dict:
    """WRITE ACTION: retrain the breast-cancer Random Forest, log a new MLflow
    run, and register it as a NEW model version. This does NOT change the
    'champion' alias or the deployed model.

    Two-step flow: call FIRST without confirm_token to get a preview and a
    token. Show the preview to the user and ask for approval. Only after the
    user says yes, call again with the SAME parameters and the token.
    Never confirm on the user's behalf.

    Args:
        n_estimators: Number of trees, 10 to 500 (default 100).
        max_depth: Max tree depth, 1 to 30, or omit for unlimited.
        confirm_token: Token from the preview step; omit on the first call.
    """
    params = {"n_estimators": n_estimators, "max_depth": max_depth}
    log.info("tool called: trigger_retrain(%s)", params)

    # 1. Validate inputs
    if not 10 <= n_estimators <= 500:
        audit("rejected", params=params, reason="n_estimators out of range")
        raise ValueError("n_estimators must be between 10 and 500.")
    if max_depth is not None and not 1 <= max_depth <= 30:
        audit("rejected", params=params, reason="max_depth out of range")
        raise ValueError("max_depth must be between 1 and 30, or omitted.")

    # 2. Read-only mode blocks everything
    if READ_ONLY:
        audit("blocked_read_only", params=params)
        return {
            "status": "BLOCKED",
            "reason": "Server is in read-only mode. Write actions are disabled.",
        }

    # 3. Step one: issue a preview and a token
    if not confirm_token:
        now = time.time()
        for t in [t for t, p in PENDING.items() if p["expires"] < now]:
            PENDING.pop(t)
        token = secrets.token_hex(4)
        PENDING[token] = {"params": params, "expires": now + TOKEN_TTL}
        audit("preview_issued", params=params, token=token)
        return {
            "status": "CONFIRMATION_REQUIRED",
            "preview": (
                f"Retrain Random Forest (n_estimators={n_estimators}, "
                f"max_depth={max_depth}) on the breast cancer data, log a new "
                "MLflow run, and register a new model version. The champion "
                "and the deployed model stay unchanged."
            ),
            "confirm_token": token,
            "expires_in_seconds": TOKEN_TTL,
            "next_step": "Ask the user to approve, then call again with the same parameters and this token.",
        }

    # 4. Step two: check the token (single use, must match the parameters)
    pending = PENDING.pop(confirm_token, None)
    if pending is None or pending["expires"] < time.time() or pending["params"] != params:
        audit("confirmation_rejected", params=params, token=confirm_token)
        return {
            "status": "REJECTED",
            "reason": "Token is invalid, expired, already used, or the parameters changed. Start again without a token.",
        }

    # 5. Execute. A write must never run without an audit record.
    if not audit("started", params=params, token=confirm_token):
        return {
            "status": "FAILED",
            "error": "The audit log is not writable, so the retrain was not started.",
        }

    cmd = [
        sys.executable,
        str(ROOT / "training" / "retrain.py"),
        "--n-estimators", str(n_estimators),
        "--register",
    ]
    if max_depth is not None:
        cmd += ["--max-depth", str(max_depth)]

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_path = logs_dir / "retrain_stdout.log"
    err_path = logs_dir / "retrain_stderr.log"
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "DO_NOT_TRACK": "true",
    }

    try:
        with open(out_path, "w", encoding="utf-8") as out, open(
            err_path, "w", encoding="utf-8"
        ) as err:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,  # do not share the MCP stdin pipe
                stdout=out,                # files instead of pipes
                stderr=err,
                timeout=120,
                cwd=str(ROOT),
                env=env,
            )
    except subprocess.TimeoutExpired:
        tail = err_path.read_text(encoding="utf-8", errors="replace")[-500:]
        audit("failed", params=params, error="timeout", stderr_tail=tail)
        return {
            "status": "FAILED",
            "error": "Retraining timed out after 120 seconds.",
            "stderr_tail": tail,
        }

    stdout = out_path.read_text(encoding="utf-8", errors="replace")
    stderr = err_path.read_text(encoding="utf-8", errors="replace")

    if proc.returncode != 0:
        audit("failed", params=params, error=stderr[-500:])
        return {"status": "FAILED", "error": stderr[-500:]}

    line = next(
        (l for l in stdout.splitlines() if l.startswith("RESULT_JSON:")), None
    )
    if line is None:
        audit("failed", params=params, error="no result line")
        return {"status": "FAILED", "error": "Training finished but returned no result."}

    result = json.loads(line[len("RESULT_JSON:"):])
    audit("completed", params=params, result=result)
    return {
        "status": "COMPLETED",
        **result,
        "note": "New version registered. Champion and the deployed model are unchanged.",
    }


@mcp.tool()
def get_logs(source: str = "audit", lines: int = 20) -> dict:
    """Read recent logs to see what happened and why something failed.

    Args:
        source: 'audit' for the record of write actions (retrain attempts),
                or 'api' for the deployed model API container's logs.
        lines: How many recent lines to return, 1 to 200 (default 20).
    """
    log.info("tool called: get_logs(%s, %s)", source, lines)
    if not 1 <= lines <= 200:
        raise ValueError("lines must be between 1 and 200.")

    if source == "audit":
        if not AUDIT_LOG.exists():
            return {"source": "audit", "entries": []}
        entries = []
        for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines()[-lines:]:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entry = {"raw": line}
            if "token" in entry:
                entry["token"] = "[redacted]"  # never expose confirmation tokens
            entries.append(entry)
        return {"source": "audit", "entries": entries}

    if source == "api":
        try:
            proc = subprocess.run(
                ["docker", "logs", "--tail", str(lines), "mlops-api"],
                capture_output=True, text=True, timeout=15,
            )
        except FileNotFoundError:
            return {
                "source": "api",
                "error": "Docker is not available in this environment, so container logs cannot be read.",
            }
        except subprocess.TimeoutExpired:
            return {"source": "api", "error": "Reading the container logs timed out."}
        if proc.returncode != 0:
            return {"source": "api", "error": proc.stderr[-500:]}
        return {"source": "api", "logs": (proc.stdout + proc.stderr)[-4000:]}

    raise ValueError("source must be 'audit' or 'api'.")


if __name__ == "__main__":
    mcp.run()  # stdio transport