import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()


def register_best(experiment, metric, model_name):
    exp = client.get_experiment_by_name(experiment)
    best = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )[0]
    mv = mlflow.register_model(f"runs:/{best.info.run_id}/model", model_name)
    client.set_registered_model_alias(model_name, "champion", mv.version)
    print(f"{model_name} v{mv.version} <- {best.info.run_name} ({metric})")


register_best("breast-cancer-classifier", "f1", "breast-cancer-model")
register_best("iris-clustering", "silhouette", "iris-cluster-model")