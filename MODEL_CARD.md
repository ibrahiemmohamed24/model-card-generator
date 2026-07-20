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

The XGBoost model is intended for binary classification of breast cancer cells based on 30 input features.

## Limitations

- Performance may vary when applied to datasets other than the provided 'Breast Cancer' dataset.
- The model's maximum depth is limited to 4 levels, which could impact its ability to capture complex patterns in the data.
- The learning rate of 0.1 might not be optimal for all subsets of the breast cancer cells.
- Limited information is available about the model's performance on imbalanced datasets.

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
