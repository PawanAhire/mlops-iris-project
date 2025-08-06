import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Set the MLflow tracking URI. Could be a local folder or a remote server.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Set the experiment name
mlflow.set_experiment("Iris Classification")

def train_models():
    """Trains, evaluates, and logs two models for the Iris dataset."""
    print("Loading data...")
    df = pd.read_csv("data/raw/iris.csv")

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Model 1: Logistic Regression ---
    with mlflow.start_run(run_name="Logistic Regression"):
        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(lr, "model")
        print(f"Logistic Regression - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # --- Model 2: Random Forest ---
    with mlflow.start_run(run_name="Random Forest"):
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(rf, "model")
        print(f"Random Forest - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

def register_best_model():
    """Finds the best run and registers the model."""
    print("Registering the best model...")
    experiment_name = "Iris Classification"
    client = mlflow.tracking.MlflowClient()

    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=client.get_experiment_by_name(experiment_name).experiment_id,
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs:
        print("No runs found. Exiting.")
        return

    best_run = runs[0]
    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "IrisClassifier"

    # Register the model
    model_version = mlflow.register_model(model_uri=best_model_uri, name=model_name)
    print(f"Model '{model_name}' registered with version {model_version.version}")

    # Transition the model to the "Production" stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    print(f"Model version {model_version.version} moved to Production.")

if __name__ == '__main__':
    train_models()
    register_best_model()