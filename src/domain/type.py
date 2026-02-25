from dataclasses import dataclass
from typing import Sequence, Union
from numpy import floating 
from numpy.typing import NDArray

Matrix = NDArray[floating]
MatrixOrSeq = Union[Matrix, Sequence[Matrix]]

@dataclass
class SimulationResult:
    """
    Minimal result container.
    The service decides what to record; these are the common trajectories.
    """
    x: Sequence[Matrix]  # typically length N+1
    y: Sequence[Matrix]  # typically length N+1
    u: Sequence[Matrix]  # typically length N
    cost: float