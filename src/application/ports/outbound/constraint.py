from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Constraint(ABC):

    @abstractmethod
    def build(self, z: Any, x0_param: Any) -> list[Any]:
        ...
