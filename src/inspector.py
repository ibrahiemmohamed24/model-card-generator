from pathlib import Path
import pickle


def inspect_model(model_path: str) -> dict:
    """Inspect a trusted saved model and return JSON-friendly metadata."""
    loaded = load_model_safe(model_path)

    if "error" in loaded:
        return {
            "model_class": None,
            "parameters": {},
            "feature_count": None,
            "output_type": None,
            "error": loaded["error"],
        }

    model = loaded["model"]
    result = {
        "model_class": type(model).__name__,
        "parameters": {},
        "feature_count": None,
        "output_type": None,
        "format": loaded["format"],
    }

    _extract_sklearn_params(model, result)
    _extract_xgboost_params(model, result)
    _extract_pytorch_params(model, result)

    
    if isinstance(model, dict) and "state_dict" in model:
        result["output_type"] = "PyTorch checkpoint"
        result["checkpoint_keys"] = list(model.keys())

    return result


def _extract_sklearn_params(model, result):
    if hasattr(model, "get_params"):
        try:
            params = model.get_params(deep=False)
            result["parameters"].update(
                {key: _to_builtin(value) for key, value in params.items()
                 if value is not None}
            )
        except Exception:
            pass

    if hasattr(model, "n_features_in_"):
        result["feature_count"] = _to_builtin(model.n_features_in_)

        if hasattr(model, "classes_"):
            class_count = len(model.classes_)
        result["output_type"] = (
            f"Classification — {class_count} classes"
        )
    else:
        try:
            from sklearn.base import is_classifier, is_regressor

            if is_classifier(model):
                result["output_type"] = "Classification"
            elif is_regressor(model):
                result["output_type"] = "Regression"

        except ImportError:
            pass


def _extract_xgboost_params(model, result):
    if hasattr(model, "get_xgb_params"):
        try:
            params = model.get_xgb_params()
            result["parameters"].update(
                {key: _to_builtin(value) for key, value in params.items()
                 if value is not None}
            )
        except Exception:
            pass

    if hasattr(model, "feature_importances_"):
        try:
            importances = model.feature_importances_.tolist()
            result["top_features"] = [
                {"feature_index": index, "importance": float(importance)}
                for index, importance in sorted(
                    enumerate(importances),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
            ]
        except Exception:
            pass


def _extract_pytorch_params(model, result):
    try:
        import torch
    except ImportError:
        return

    if not isinstance(model, torch.nn.Module):
        return

    result["model_class"] = type(model).__name__

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    result["parameters"].update({
        "total_params": total_params,
        "trainable_params": trainable_params,
    })

    if result["output_type"] is None:
        result["output_type"] = "Neural Network"


def load_model_safe(model_path: str) -> dict:
    """
    Load ONLY trusted model files.
    pickle, joblib, and full PyTorch checkpoints may execute arbitrary code.
    """
    path = Path(model_path)
    extension = path.suffix.lower()

    supported = {".pkl", ".pickle", ".joblib", ".pt", ".pth"}

    if not path.is_file():
        return {"error": f"Model file not found: {path}"}

    if extension not in supported:
        return {
            "error": (
                f"Unsupported format: {extension}. "
                f"Supported: {sorted(supported)}"
            )
        }

    try:
        if extension == ".joblib":
            import joblib
            return {"model": joblib.load(path), "format": "joblib"}

        if extension in {".pt", ".pth"}:
            import torch
            return {
                "model": torch.load(path, map_location="cpu", weights_only=False),
                "format": "pytorch",
            }

        with path.open("rb") as file:
            return {"model": pickle.load(file), "format": "pickle"}

    except Exception as error:
        return {"error": f"Could not load model: {error}"}


def _to_builtin(value):
    """Convert common NumPy-like scalar values into JSON-compatible values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]

    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}

    return str(value)