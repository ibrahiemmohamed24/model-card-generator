import json
import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


random_state = 42
test_size = 0.2
n_estimators = 100
max_depth = 5

output_directory = Path(__file__).resolve().parent
model_path = output_directory / "sample_model.pkl"
metrics_path = output_directory / "sample_metrics.json"

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_value = accuracy_score(y_test, y_pred)
f1_value = f1_score(y_test, y_pred, average="weighted")

metrics = {
    "accuracy": float(accuracy_value),
    "f1_score_weighted": float(f1_value),
}

print(f"Accuracy: {accuracy_value:.4f}")
print(f"F1 Score: {f1_value:.4f}")

with model_path.open("wb") as file:
    pickle.dump(model, file)

with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print(f"Model saved to {model_path}")
print(f"Metrics saved to {metrics_path}")