import json
import sys
from pathlib import Path
from typing import Optional

from src.generator import generate_model_card
from src.inspector import inspect_model
from src.parser import extract_from_code


def _infer_metrics_path(code_path: str) -> Path:
    code_file = Path(code_path)
    stem = code_file.stem

    candidates = [
        code_file.with_name(f"{stem}_metrics.json"),
    ]

    if stem.startswith("sample_"):
        short_name = stem.removeprefix("sample_")
        candidates.append(
            code_file.with_name(f"{short_name}_metrics.json")
        )

    if stem == "sample_model":
        candidates.append(
            code_file.with_name("sample_metrics.json")
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


def run(
    code_path: str,
    model_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    output_path: str = "MODEL_CARD.md",
    print_card: bool = True,
) -> str:
    print("Extracting metadata from code...")
    metadata = extract_from_code(code_path)

    if model_path:
        model_file = Path(model_path)

        if model_file.is_file():
            print("Inspecting model file...")
            inspector_result = inspect_model(str(model_file))

            if "error" in inspector_result:
                print(f"Warning: {inspector_result['error']}")
            else:
                metadata.update(inspector_result)
        else:
            print(f"Warning: Model file not found: {model_file}")

    if metrics_path is None:
        metrics_file = _infer_metrics_path(code_path)
    else:
        metrics_file = Path(metrics_path)

    if metrics_file.is_file():
        print(f"Loading evaluation metrics from {metrics_file}...")

        try:
            runtime_metrics = json.loads(
                metrics_file.read_text(encoding="utf-8")
            )

            if isinstance(runtime_metrics, dict):
                metadata.setdefault("metrics", {})
                metadata["metrics"].update(runtime_metrics)
            else:
                print("Warning: Metrics file must contain a JSON object.")

        except (OSError, json.JSONDecodeError) as error:
            print(f"Warning: Could not load metrics: {error}")
    else:
        print(f"Warning: Metrics file not found: {metrics_file}")

    metadata["source_code_path"] = Path(code_path).as_posix()
    metadata["model_file_path"] = (
        Path(model_path).as_posix()
        if model_path
        else None
    )
    metadata["metrics_file_path"] = metrics_file.as_posix()

    print("Generating model card with Hermes3...")
    card = generate_model_card(
        metadata,
        output_path=output_path,
    )

    print(f"\nDone! {output_path} generated successfully.")
    if print_card:
        print("-" * 40)
        print(card)

    return card


if __name__ == "__main__":
    code_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "tests/sample_model.py"
    )

    model_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "tests/sample_model.pkl"
    )

    metrics_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else None
    )

    run(code_path, model_path, metrics_path)