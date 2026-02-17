from typing import Sequence, Union
from numpy import floating 
from numpy.typing import NDArray

Matrix = NDArray[floating]
MatrixOrSeq = Union[Matrix, Sequence[Matrix]]