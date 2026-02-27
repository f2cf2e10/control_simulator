import numpy as np

experiment1 = {
    "A": np.array([[1.0, 0.1],
                   [0.0, 1.0]]),
    "B": np.array([[0.0],
                   [1.0]]),
    "C": np.array([[1.0, 0.0]]),      # measure position only
    "Q": np.diag([10.0, 1.0]),
    "R": np.array([[1]]),
    "Qn": np.diag([10.0, 50.0]),
    "SigmaController": np.array([[0.1, 0.],
                                 [0., 0.1]]),   # process noise covariance
    "GammaController": np.array([[1.]]),      # measurement noise covariance
    "SigmaPlant": np.array([[0.2, 0.1],
                            [0., 0.2]]),   # process noise covariance
    "GammaPlant": np.array([[0.1]]),      # measurement noise covariance
    "x0_cov": np.diag([1.0, 1.0]),    # initial estimation covariance P0
    "x0_mean": np.array([[10.0],
                         [0.0]])
}
