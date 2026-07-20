# Model Card Generator

Auto-generates professional Model Cards for ML models using a local LLM.

## What it does

Point it at a training script and a saved model — it extracts hyperparameters, metrics, train/test split, and random seed automatically, then uses Hermes3 (via Ollama) to write the Limitations and Intended Use sections.

## Supported Models

- scikit-learn (RandomForest, SVM, LinearModels, DecisionTree)
- XGBoost
- PyTorch (coming soon)

## Requirements

- Python 3.8+
- Ollama running locally with hermes3 model

```bash
ollama pull hermes3
```

## Installation

```bash
git clone https://github.com/your-username/model-card-generator
cd model-card-generator
pip install -r requirements.txt
```

## Usage

```bash
# sklearn
python cli.py generate tests/sample_model.py -m tests/sample_model.pkl

# XGBoost
python cli.py generate tests/sample_xgboost.py -m tests/sample_xgboost.pkl --metrics tests/xgboost_metrics.json

# Custom output path
python cli.py generate training.py -m model.pkl -o docs/MODEL_CARD.md
```

## Output

Generates a `MODEL_CARD.md` containing:

- Model details (type, class, dataset, feature count)
- Hyperparameters table
- Evaluation metrics table
- Intended Use (LLM-generated)
- Limitations (LLM-generated)
- Reproducibility steps

## Project Structure
model-card-generator/
├── src/
│ ├── parser.py # AST + regex extraction from training code
│ ├── inspector.py # Model file inspection
│ └── generator.py # LLM call + markdown assembly
├── tests/
│ ├── sample_model.py # sklearn demo
│ └── sample_xgboost.py # XGBoost demo
├── cli.py # CLI interface
├── main.py # Programmatic entry point
└── index.html # Project landing page


## Stack

- Python 3.14
- Ollama + Hermes3 (local LLM)
- scikit-learn
- XGBoost