import logging
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient
from mcp.server.fastmcp import FastMCP

# IMPORTANT: stdio servers must never print() to stdout.
# stdout is reserved for the MCP protocol, so log to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("mlops-copilot")

DB_PATH = Path(__file__).resolve().parent.parent / "mlflow.db"
client = MlflowClient(tracking_uri=f"sqlite:///{DB_PATH.as_posix()}")

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
        top_n: How many top runs to return (default 5).
        higher_is_better: True for accuracy/f1/silhouette, False for error metrics like rmse.
    """
    log.info("tool called: compare_runs(%s, %s)", experiment_name, metric)
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


if __name__ == "__main__":
    mcp.run()  # stdio transport