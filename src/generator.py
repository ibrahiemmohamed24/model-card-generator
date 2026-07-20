import json
import os
from datetime import date
from pathlib import Path

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
HERMES_MODEL = os.getenv("HERMES_MODEL", "hermes3")


def generate_model_card(
    metadata: dict,
    output_path: str = "MODEL_CARD.md",
) -> str:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    llm_sections = _call_hermes(metadata)
    card = _build_card(metadata, llm_sections)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(card, encoding="utf-8")

    return card


def _call_hermes(metadata: dict) -> dict:
    prompt = f"""
You are an ML documentation expert.

Use the provided model metadata to write these sections in English:

1. Intended Use:
   Write 2-3 sentences describing the model's intended purpose.

2. Limitations:
   Write 3-4 concise limitations.

Rules:
- Do not invent datasets, metrics, or performance claims.
- Describe the model purpose without subjective words such as accurate, strong, high, good, or excellent.
- Mention metric values only as reported numbers.
- Limitations must be directly supported by the provided metadata.
- When random_seed is provided, do not claim run-to-run randomness under identical conditions.
- Mention the dataset only when dataset_name is available.
- Clearly state when information is unavailable.
- Return valid JSON only.
- Do not include markdown code fences or additional text.

Model metadata:
{json.dumps(metadata, indent=2, default=str)}

Required JSON format:
{{
  "intended_use": "text",
  "limitations": [
    "limitation one",
    "limitation two",
    "limitation three"
  ]
}}
""".strip()

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": HERMES_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=180,
        )
        response.raise_for_status()

        response_data = response.json()
        raw_content = response_data.get("response")

        if not raw_content:
            raise ValueError("Hermes returned an empty response")

        sections = json.loads(raw_content)

        if not isinstance(sections, dict):
            raise ValueError("Hermes response must be a JSON object")

        intended_use = sections.get("intended_use", "Not specified.")
        limitations = sections.get("limitations", [])

        if not isinstance(intended_use, str):
            intended_use = str(intended_use)

        if not isinstance(limitations, list):
            limitations = [str(limitations)]

        limitations = [
            str(item).strip()
            for item in limitations
            if str(item).strip()
        ]

        return {
            "intended_use": intended_use.strip(),
            "limitations": limitations,
        }

    except (
        requests.RequestException,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        return {
            "intended_use": (
                "Intended use could not be generated because "
                "the Hermes request failed."
            ),
            "limitations": [
                f"Hermes agent error: {error}",
            ],
        }


def _build_card(metadata: dict, llm_sections: dict) -> str:
    today = date.today().isoformat()

    model_type = (
        metadata.get("model_type")
        or metadata.get("model_class")
        or "Unknown Model"
    )
    model_class = metadata.get("model_class") or "Not specified"
    dataset_name = metadata.get("dataset_name") or "Not specified"
    hyperparameters = {}

    inspector_parameters = metadata.get("parameters", {})
    parser_parameters = metadata.get("hyperparameters", {})

    if isinstance(inspector_parameters, dict):
        hyperparameters.update(inspector_parameters)

    if isinstance(parser_parameters, dict):
        hyperparameters.update(parser_parameters)

    metrics = metadata.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    test_size = metadata.get("train_test_split")

    if (
        isinstance(test_size, (int, float))
        and not isinstance(test_size, bool)
        and 0 <= test_size <= 1
    ):
        train_test_split = (
            f"{1 - test_size:.0%} train / {test_size:.0%} test"
        )
    else:
        train_test_split = test_size or "Not specified"
    random_seed = metadata.get(
        "random_seed",
        "Not specified",
    )
    feature_count = metadata.get(
        "feature_count",
        "Not specified",
    )
    output_type = metadata.get(
        "output_type",
        "Not specified",
    )

    intended_use = llm_sections.get(
        "intended_use",
        "Not specified.",
    )
    limitations = llm_sections.get("limitations", [])

    hyperparameters_rows = _build_table_rows(hyperparameters)
    formatted_metrics = {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in metrics.items()
    }
    metrics_rows = _build_table_rows(formatted_metrics)

    if limitations:
        limitations_section = "\n".join(
            f"- {_format_markdown_value(item)}"
            for item in limitations
        )
    else:
        limitations_section = "- Not specified."

    source_code_path = (
        metadata.get("source_code_path")
        or "tests/sample_model.py"
    )
    model_file_path = metadata.get("model_file_path")
    metrics_file_path = metadata.get("metrics_file_path")

    training_command = (
        f"python {json.dumps(source_code_path)}"
    )

    generation_arguments = [source_code_path]

    if model_file_path:
        generation_arguments.append(model_file_path)

    if metrics_file_path:
        generation_arguments.append(metrics_file_path)

    generation_command = "python main.py " + " ".join(
        json.dumps(str(argument))
        for argument in generation_arguments
    )
    return f"""# Model Card: {_format_markdown_value(model_type)}

## Model Details

| Field | Value |
|---|---|
| Model Type | {_format_markdown_value(model_type)} |
| Model Class | {_format_markdown_value(model_class)} |
| Dataset | {_format_markdown_value(dataset_name)} |
| Output Type | {_format_markdown_value(output_type)} |
| Feature Count | {_format_markdown_value(feature_count)} |
| Random Seed | {_format_markdown_value(random_seed)} |
| Train/Test Split | {_format_markdown_value(train_test_split)} |
| Date Generated | {today} |

## Hyperparameters

| Parameter | Value |
|---|---|
{hyperparameters_rows}

## Evaluation Metrics

| Metric | Value |
|---|---|
{metrics_rows}

## Intended Use

{_format_markdown_value(intended_use)}

## Limitations

{limitations_section}

## Reproducibility

1. Install the project dependencies:

~~~bash
pip install -r requirements.txt
~~~

2. Run the training script:

~~~bash
{training_command}
~~~

3. Generate the model card:

~~~bash
{generation_command}
~~~
"""


def _build_table_rows(values: dict) -> str:
    if not values:
        return "| Not specified | Not specified |"

    rows = []

    for key, value in sorted(
        values.items(),
        key=lambda item: str(item[0]),
    ):
        formatted_key = _format_markdown_value(key)
        formatted_value = _format_markdown_value(value)
        rows.append(f"| {formatted_key} | {formatted_value} |")

    return "\n".join(rows)


def _format_markdown_value(value) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        value = json.dumps(
            value,
            default=str,
            ensure_ascii=False,
        )

    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )