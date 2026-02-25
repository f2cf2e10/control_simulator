import numpy as np

plant1 = {
    "A": np.array([[1.0, 0.1],
                   [0.0, 1.0]]),
    "B": np.array([[0.0],
                   [1.0]]),
    "C": np.array([[1.0, 0.0]]),      # measure position only
    "Q": np.diag([10.0, 1.0]),
    "R": np.array([[1]]),
    "Qn": np.diag([10.0, 50.0]),
    "Sigma": np.diag([1e-1, 1e-1]),   # process noise covariance
    "Gamma": np.array([[1e-1]]),      # measurement noise covariance
    "x0_cov": np.diag([1.0, 1.0]),    # initial estimation covariance P0
    "x0_mean": np.array([[10.0],
                         [0.0]])
}
