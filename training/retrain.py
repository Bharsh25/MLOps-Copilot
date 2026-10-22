import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
MODEL_NAME = "breast-cancer-model"

p = argparse.ArgumentParser()
p.add_argument("--n-estimators", type=int, default=100)
p.add_argument("--max-depth", type=int, default=None)
p.add_argument("--register", action="store_true")
a = p.parse_args()

mlflow.set_experiment("breast-cancer-classifier")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name=f"retrain_rf_n{a.n_estimators}_d{a.max_depth}") as run:
    model = RandomForestClassifier(
        n_estimators=a.n_estimators, max_depth=a.max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

    mlflow.set_tag("triggered_by", "mcp-trigger_retrain")
    mlflow.log_params(
        {
            "model_type": "random_forest",
            "n_estimators": a.n_estimators,
            "max_depth": str(a.max_depth),
        }
    )
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    result = {"run_id": run.info.run_id, "metrics": metrics}
    if a.register:
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
        result["registered_version"] = str(mv.version)

print("RESULT_JSON:" + json.dumps(result))