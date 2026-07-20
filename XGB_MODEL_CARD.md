# Model Card: XGBoost

## Model Details

| Field | Value |
|---|---|
| Model Type | XGBoost |
| Model Class | XGBClassifier |
| Dataset | Breast Cancer |
| Output Type | Classification — 2 classes |
| Feature Count | 30 |
| Random Seed | 42 |
| Train/Test Split | 80% train / 20% test |
| Date Generated | 2026-07-21 |

## Hyperparameters

| Parameter | Value |
|---|---|
| enable_categorical | False |
| eval_metric | logloss |
| learning_rate | 0.1 |
| max_depth | 4 |
| missing | nan |
| n_estimators | 200 |
| objective | binary:logistic |
| random_state | 42 |
| verbosity | 0 |

## Evaluation Metrics

| Metric | Value |
|---|---|
| accuracy | 0.9561 |
| f1_score_weighted | 0.956 |

## Intended Use

The XGBoost model is intended for binary classification of the Breast Cancer dataset, distinguishing between two classes.

## Limitations

- Limited to the specific Breast Cancer dataset and may not generalize well to other datasets without retraining.
- Performance metrics are based on a single train-test split with a fixed random seed, and performance may vary in real-world applications.
- The model has a fixed depth of 4 and a learning rate of 0.1, which could limit its ability to capture complex patterns in the data.
- Only the top 5 features are considered by importance; other potentially relevant features may be ignored.

## Reproducibility

1. Install the project dependencies:

~~~bash
pip install -r requirements.txt
~~~

2. Run the training script:

~~~bash
python "tests/sample_xgboost.py"
~~~

3. Generate the model card:

~~~bash
python main.py "tests/sample_xgboost.py" "tests/sample_xgboost.pkl" "tests/xgboost_metrics.json"
~~~
