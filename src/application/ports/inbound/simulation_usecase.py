from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.type import Matrix, SimulationResult


class SimulationUseCase(ABC):
    """
    Inbound port.
    """
    @abstractmethod
    def execute(self) -> SimulationResult:
        ...
