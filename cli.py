import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from main import run


VERSION = "1.0.0"
MODEL_EXTENSIONS = (
    ".pkl",
    ".pickle",
    ".joblib",
    ".pt",
    ".pth",
)


def _existing_file(value: str) -> Path:
    path = Path(value)

    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"File not found: {path}"
        )

    return path


def _infer_model_path(code_path: Path) -> Optional[Path]:
    for extension in MODEL_EXTENSIONS:
        candidate = code_path.with_suffix(extension)

        if candidate.is_file():
            return candidate

    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-card-generator",
        description=(
            "Generate ML model cards from training code, "
            "saved models, and evaluation metrics."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a model card.",
    )

    generate_parser.add_argument(
        "code",
        type=_existing_file,
        help="Path to the Python training script.",
    )

    model_group = generate_parser.add_mutually_exclusive_group()

    model_group.add_argument(
        "-m",
        "--model",
        type=_existing_file,
        help="Path to the saved model file.",
    )

    model_group.add_argument(
        "--skip-model",
        action="store_true",
        help="Generate the card without inspecting a model file.",
    )

    generate_parser.add_argument(
        "--metrics",
        type=_existing_file,
        help=(
            "Path to the metrics JSON file. "
            "Detected automatically when omitted."
        ),
    )

    generate_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("MODEL_CARD.md"),
        help="Output Markdown path. Default: MODEL_CARD.md",
    )

    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )

    generate_parser.add_argument(
        "--no-print-card",
        action="store_true",
        help="Do not print the generated card to the terminal.",
    )

    return parser


def _generate(args: argparse.Namespace) -> int:
    code_path: Path = args.code
    output_path: Path = args.output

    if code_path.suffix.lower() != ".py":
        raise ValueError(
            "The training code must be a Python file."
        )

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --force to overwrite it."
        )

    if args.skip_model:
        model_path = None
    elif args.model:
        model_path = args.model
    else:
        model_path = _infer_model_path(code_path)

        if model_path:
            print(f"Detected model file: {model_path}")
        else:
            print(
                "Warning: No matching model file was detected. "
                "Continuing with code metadata only."
            )

    run(
        code_path=str(code_path),
        model_path=str(model_path) if model_path else None,
        metrics_path=(
            str(args.metrics)
            if args.metrics
            else None
        ),
        output_path=str(output_path),
        print_card=not args.no_print_card,
    )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            return _generate(args)

        parser.error(f"Unknown command: {args.command}")

    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 130

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())