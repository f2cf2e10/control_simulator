from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.domain.type import Matrix


@dataclass
class ChanceConstraintSpec:
    Cprev: Matrix
    Ccurr: Matrix
    b: Matrix
    mean_selector: Optional[Callable[[Any], Any]] = None
    step_start: int = 1
    step_stop: Optional[int] = None  # exclusive


@dataclass
class AncillaryControlLaw:
    transform: Callable[[Matrix, Matrix], Matrix]

    def __call__(self, v0: Matrix, y_k: Matrix) -> Matrix:
        return self.transform(v0, y_k)
