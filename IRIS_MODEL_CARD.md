# Model Card: Ensemble Model (sklearn)

## Model Details

| Field | Value |
|---|---|
| Model Type | Ensemble Model (sklearn) |
| Model Class | RandomForestClassifier |
| Dataset | Iris |
| Output Type | Classification — 3 classes |
| Feature Count | 4 |
| Random Seed | 42 |
| Train/Test Split | 80% train / 20% test |
| Date Generated | 2026-07-21 |

## Hyperparameters

| Parameter | Value |
|---|---|
| bootstrap | True |
| ccp_alpha | 0.0 |
| criterion | gini |
| max_depth | 5 |
| max_features | sqrt |
| min_impurity_decrease | 0.0 |
| min_samples_leaf | 1 |
| min_samples_split | 2 |
| min_weight_fraction_leaf | 0.0 |
| n_estimators | 100 |
| oob_score | False |
| random_state | 42 |
| verbose | 0 |
| warm_start | False |

## Evaluation Metrics

| Metric | Value |
|---|---|
| accuracy | 0.9333 |
| f1_score_weighted | 0.9333 |

## Intended Use

This model is designed to classify Iris flowers into three categories based on their sepal length, sepal width, petal length, and petal width features.

## Limitations

- The model's accuracy is dependent on the quality of the input data and may not perform well if presented with unfamiliar or mislabeled samples.
- As the model was trained on the Iris dataset, it may have limited applicability to other flower species or classifications.
- The random forest classifier relies on a subset of the training data for each tree, which could lead to variations in performance when retrained with different seeds.

## Reproducibility

1. Install the project dependencies:

~~~bash
pip install -r requirements.txt
~~~

2. Run the training script:

~~~bash
python "tests/sample_model.py"
~~~

3. Generate the model card:

~~~bash
python main.py "tests/sample_model.py" "tests/sample_model.pkl" "tests/sample_metrics.json"
~~~
