from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from src.domain.type import Matrix


@dataclass
class SimulationResult:
    """
    Minimal result container.
    The service decides what to record; these are the common trajectories.
    """
    x: Sequence[Matrix]  # typically length N+1
    y: Sequence[Matrix]  # typically length N+1
    u: Sequence[Matrix]  # typically length N


class SimulationUseCase(ABC):
    """
    Inbound port.
    """
    @abstractmethod
    def execute(self) -> SimulationResult:
        ...

