import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-clustering")

X, _ = load_iris(return_X_y=True)

for k in [2, 3, 4, 5]:
    with mlflow.start_run(run_name=f"kmeans_k{k}"):
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        mlflow.log_param("n_clusters", k)
        mlflow.log_metric("silhouette", silhouette_score(X, model.labels_))
        mlflow.sklearn.log_model(model, name="model")