import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
MODEL_NAME = "breast-cancer-model"

client = MlflowClient()
mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
run = client.get_run(mv.run_id)

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
joblib.dump(model, ROOT / "api" / "model.joblib")

meta = {
    "model_name": MODEL_NAME,
    "model_version": str(mv.version),
    "run_id": mv.run_id,
    "metrics": run.data.metrics,
}
(ROOT / "api" / "model_meta.json").write_text(json.dumps(meta, indent=2))
print("Exported:", meta)