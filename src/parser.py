import ast
import re


def extract_from_code(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    result = {
        "hyperparameters": {},
        "metrics": {},
        "train_test_split": None,
        "random_seed": None,
        "model_type": None,
        "dataset_name": None,
        "imports": [],
    }

    _extract_imports(tree, result)
    _extract_dataset(tree, result)
    _extract_assignments(tree, result)
    _extract_metrics(source, result)

    return result


def _extract_dataset(tree, result):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        else:
            continue

        if function_name.startswith(("load_", "fetch_")):
            dataset_name = function_name.split("_", 1)[1]
            result["dataset_name"] = (
                dataset_name.replace("_", " ").title()
            )
            return
def _extract_imports(tree, result):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
                _detect_model_type(alias.name, result)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result["imports"].append(node.module)
                _detect_model_type(node.module, result)


def _detect_model_type(module: str, result):
    model_map = {
        "sklearn.ensemble": "Ensemble Model (sklearn)",
        "sklearn.linear_model": "Linear Model (sklearn)",
        "sklearn.svm": "Support Vector Machine (sklearn)",
        "sklearn.tree": "Decision Tree (sklearn)",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "torch": "PyTorch Neural Network",
        "tensorflow": "TensorFlow Neural Network",
    }
    for key, value in model_map.items():
        if key in module:
            result["model_type"] = value


def _extract_assignments(tree, result):
    param_keywords = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "epochs",
        "batch_size",
        "hidden_size",
        "dropout",
        "num_leaves",
        "min_samples_split",
        "C",
        "kernel",
        "gamma",
    ]
    seed_keywords = ["random_state", "seed", "random_seed"]
    split_keywords = ["test_size", "train_size"]

    assigned_values = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        value = _get_value(node.value)

        for target in node.targets:
            if isinstance(target, ast.Name):
                assigned_values[target.id] = value

    def resolve_value(node):
        if isinstance(node, ast.Name) and node.id in assigned_values:
            return assigned_values[node.id]

        return _get_value(node)

    def store_value(name, value):
        if name in param_keywords:
            result["hyperparameters"][name] = value

        elif name in seed_keywords:
            result["random_seed"] = value

        elif name in split_keywords:
            if name == "test_size" or result["train_test_split"] is None:
                result["train_test_split"] = value

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = resolve_value(node.value)

            for target in node.targets:
                if isinstance(target, ast.Name):
                    store_value(target.id, value)

        elif isinstance(node, ast.keyword) and node.arg:
            value = resolve_value(node.value)
            store_value(node.arg, value)


def _extract_metrics(source: str, result):
    patterns = {
    "accuracy": r"accuracy[_\s]*[=:]\s*([\d.]+)",
    "f1_score": r"f1[_\s]*score[_\s]*[=:]\s*([\d.]+)",
    "auc": r"auc[_\s]*[=:]\s*([\d.]+)",
    "precision": r"precision[_\s]*[=:]\s*([\d.]+)",
    "recall": r"recall[_\s]*[=:]\s*([\d.]+)",
    "loss": r"loss[_\s]*[=:]\s*([\d.]+)",
    "rmse": r"rmse[_\s]*[=:]\s*([\d.]+)",
    "mae": r"mae[_\s]*[=:]\s*([\d.]+)",
    }
    for metric, pattern in patterns.items():
        match = re.search(pattern, source, re.IGNORECASE)
        if match:
            result["metrics"][metric] = float(match.group(1))


def _get_value(node):
    """Extract a safe, serializable representation from an AST node."""

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.List):
        return [_get_value(element) for element in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_get_value(element) for element in node.elts)

    if isinstance(node, ast.Set):
        return {_get_value(element) for element in node.elts}

    if isinstance(node, ast.Dict):
        result = {}

        for key, value in zip(node.keys, node.values):
            # key=None represents dictionary unpacking: {**other_dict}
            if key is None:
                return ast.unparse(node)

            parsed_key = _get_value(key)
            parsed_value = _get_value(value)

            try:
                result[parsed_key] = parsed_value
            except TypeError:
                return ast.unparse(node)

        return result

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _get_value(node.operand)

        if isinstance(value, (int, float, complex)):
            return -value if isinstance(node.op, ast.USub) else +value

        return ast.unparse(node)

    if isinstance(node, ast.BinOp):
        left = _get_value(node.left)
        right = _get_value(node.right)

        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            operators = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.FloorDiv: lambda a, b: a // b,
                ast.Mod: lambda a, b: a % b,
                ast.Pow: lambda a, b: a ** b,
            }

            operation = operators.get(type(node.op))
            if operation:
                try:
                    return operation(left, right)
                except (ArithmeticError, OverflowError):
                    pass

        return ast.unparse(node)

    if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
        return ast.unparse(node)

    return None