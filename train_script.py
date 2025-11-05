import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🚀 Running training script...")

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Model trained successfully. Accuracy: {acc:.2f}")

# Log to MLflow (DagsHub)
mlflow.set_tracking_uri("https://dagshub.com/aakanshadijendra3-pixel/mlops-demo.mlflow")
mlflow.set_experiment("CI-CD-Training")
with mlflow.start_run(run_name="CI-CD-Pipeline-Run"):
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", acc)
    import joblib
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl", artifact_path="model")
    

print("🏁 MLflow logging complete!")
