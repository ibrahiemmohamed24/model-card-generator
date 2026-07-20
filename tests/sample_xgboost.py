import pickle
import json
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

random_state = 42
test_size = 0.2
n_estimators = 200
max_depth = 4
learning_rate = 0.1

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

model = xgb.XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=random_state,
    eval_metric="logloss",
    verbosity=0,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

with open("tests/sample_xgboost.pkl", "wb") as f:
    pickle.dump(model, f)

metrics = {"accuracy": round(accuracy, 4), "f1_score_weighted": round(f1, 4)}
with open("tests/xgboost_metrics.json", "w") as f:
    json.dump(metrics, f)

print("Model saved to tests/sample_xgboost.pkl")