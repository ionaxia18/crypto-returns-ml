"""The student-facing model interface.

Students implement exactly one class with two methods and hand its location
to the harness as ``path/to/module.py:ClassName``:

    class MyModel:
        def fit(self, train):            # train: f522kit.window.TrainWindow
            ...
        def predict(self, X):            # X: float32 [n, K] — K = the
            return scores                #   dataset's feature count

A fresh instance is constructed for every refit anchor (no state carries
across anchors; warm starts are out of contract). ``predict`` is
called once per scored date on that date's flattened rows and must return
one finite score per row. Scores may be on any scale — the evaluator's COR
is scale-invariant — but APS/AR pass through the frozen alpha clip, so
predictions should live in return units.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .window import TrainWindow


@runtime_checkable
class StudentModel(Protocol):
    def fit(self, train: TrainWindow) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class AdapterError(RuntimeError):
    """Raised when a student model spec cannot be loaded or misbehaves."""


def load_model_factory(spec: str):
    """Resolve ``path/to/module.py:ClassName`` into a zero-arg factory."""
    if ":" not in spec:
        raise AdapterError(
            f"Model spec {spec!r} must look like path/to/module.py:ClassName"
        )
    path_part, class_name = spec.rsplit(":", 1)
    path = Path(path_part)
    if not path.is_file():
        raise AdapterError(f"Model file not found: {path}")
    module_name = f"_f522kit_student_{path.stem}"
    loader_spec = importlib.util.spec_from_file_location(module_name, path)
    if loader_spec is None or loader_spec.loader is None:
        raise AdapterError(f"Cannot import model module from {path}")
    module = importlib.util.module_from_spec(loader_spec)
    sys.modules[module_name] = module
    loader_spec.loader.exec_module(module)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise AdapterError(f"{path} has no class named {class_name!r}") from exc

    def factory() -> StudentModel:
        model = cls()
        if not isinstance(model, StudentModel):
            raise AdapterError(
                f"{class_name} must define fit(train) and predict(X) methods"
            )
        return model

    return factory


def check_predictions(scores: np.ndarray, n_rows: int, context: str) -> np.ndarray:
    """Validate one date's predictions and cast to float32."""
    scores = np.asarray(scores)
    if scores.shape != (n_rows,):
        raise AdapterError(
            f"{context}: predict returned shape {scores.shape}, expected ({n_rows},)"
        )
    if not np.all(np.isfinite(scores)):
        raise AdapterError(f"{context}: predict returned non-finite values")
    return scores.astype(np.float32, copy=False)
