import numpy as np

marginally_stable_system = {
    "A": np.array([[1.0, 0.1],
                   [0.0, 1.0]]),
    "B": np.array([[0.0],
                   [1.0]]),
    "C": np.array([[1.0, 0.0]]),
    "Q": np.diag([1.0, 0.1]),
    "R": np.array([[1]]),
    "Qn": np.diag([1.0, 5.0]),
    "SigmaWrongGuess": np.array([[0.06, -0.1],
                                 [-0.1, 0.30]]),
    "GammaWrongGuess": np.array([[0.03]]),
    "SigmaPlant": np.array([[0.2, 0.08],
                            [0.08, 0.18]]),
    "GammaPlant": np.array([[0.1]]),
    "x0_cov": np.diag([1.0, 1.0]),
    "x0_mean": np.array([[10.0],
                         [0.0]])
}

unstable_system = {
    "A": np.array([[1.05, 0.1],
                   [0.0, 0.98]]),
    "B": np.array([[0.0],
                   [1.0]]),
    "C": np.array([[1.0, 0.0]]),
    "Q": np.diag([1.0, 0.1]),
    "R": np.array([[1]]),
    "Qn": np.diag([1.0, 5.0]),
    "SigmaWrongGuess": np.array([[0.06, -0.1],
                                 [-0.1, 0.30]]),
    "GammaWrongGuess": np.array([[0.03]]),
    "SigmaPlant": np.array([[0.2, 0.08],
                            [0.08, 0.18]]),
    "GammaPlant": np.array([[0.1]]),
    "x0_cov": np.diag([1.0, 1.0]),
    "x0_mean": np.array([[10.0],
                         [0.0]])
}