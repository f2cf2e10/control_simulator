import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles

from src.infrastructure.adapters.outbound.controllers.mpc.constraint.chance_constraint import ChanceConstraintNew
from src.infrastructure.adapters.outbound.controllers.mpc.constraint.input_tightening_constraint import InputTighteningConstraint
from src.infrastructure.adapters.outbound.controllers.mpc.constraint.uniform_sum_quantile import (
    quantile_sum_uniform_symmetric,
)
from src.infrastructure.adapters.outbound.plants.linear_plant import LinearPlant
from src.infrastructure.adapters.outbound.noise_samplers import UniformNoise, ZeroNoise
from src.infrastructure.adapters.outbound.cost import Quadratic
from src.infrastructure.adapters.outbound.controllers.mpc.tightened_smpc import TightenedTubeSmpc


def _chain_quantiles(Acl: np.ndarray, Bw: np.ndarray, C: np.ndarray, gamma: float, epsilon: float, a: float, p: int) -> np.ndarray:
    n = int(Acl.shape[0])
    d = int(C.shape[0])
    g_mat = (1.0 - gamma) * np.eye(n) - Acl
    q = np.zeros((p, d), dtype=float)

    for i in range(1, p + 1):
        for j in range(d):
            c = C[j:j + 1, :]
            coeffs = []
            if i > 1:
                for t in range(i - 1):
                    power = i - 2 - t
                    a_pow = np.linalg.matrix_power(Acl, power)
                    coeffs.extend((c @ a_pow @ g_mat @ Bw).reshape(-1).tolist())
            coeffs.extend((-(c @ Bw)).reshape(-1).tolist())
            q[i - 1, j] = quantile_sum_uniform_symmetric(1.0 - epsilon, a, np.asarray(coeffs))
    return q


def _velocity_quantiles(Acl: np.ndarray, Bw: np.ndarray, epsilon: float, a: float, p: int) -> np.ndarray:
    n = int(Acl.shape[0])
    e_vel = np.zeros((2, n))
    e_vel[0, 2] = 1.0
    e_vel[1, 3] = 1.0
    s_signs = np.array([[1.0, 1.0],
                        [1.0, -1.0],
                        [-1.0, 1.0],
                        [-1.0, -1.0]])

    q = np.zeros((p, 4), dtype=float)
    eps_face = epsilon / 4.0
    for i in range(1, p + 1):
        for face in range(4):
            s = s_signs[face:face + 1, :].T
            g = e_vel.T @ s
            coeffs = []
            for t in range(i):
                power = i - 1 - t
                a_pow = np.linalg.matrix_power(Acl, power)
                coeffs.extend((g.T @ a_pow @ Bw).reshape(-1).tolist())
            q[i - 1, face] = quantile_sum_uniform_symmetric(1.0 - eps_face, a, np.asarray(coeffs))
    return q


def _precompute_error_covariances(Acl: np.ndarray, Bw: np.ndarray, Sigma_w: np.ndarray, H: int) -> list[np.ndarray]:
    # Matches robust_setdiff.m:
    # Sige{1} = Bw*Sigw*Bw', Sige{k+1} = Acl*Sige{k}*Acl' + Bw*Sigw*Bw'
    bw_cov = Bw @ Sigma_w @ Bw.T
    sig = bw_cov.copy()
    seq = []
    for _ in range(H):
        seq.append(sig)
        sig = Acl @ sig @ Acl.T + bw_cov
    return seq


def _alpha_metrics_from_solution(
    z_solution: np.ndarray,
    x_curr: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    K: np.ndarray,
    sige_seq: list[np.ndarray],
    H: int,
    Hcbf: int,
) -> tuple[float, float, float, float, float]:
    n = int(x_curr.size)
    m = int(R.shape[0])
    nx = H * n

    x_pred = z_solution[:nx].reshape(H, n)     # x1..xH
    u_pred = z_solution[nx:].reshape(H, m)     # u0..u_{H-1}

    # L(k) in centralopt.m uses z(:,k), where z(:,1)=x_curr.
    L = np.zeros(H, dtype=float)
    for k in range(H):
        xk = x_curr if k == 0 else x_pred[k - 1]
        uk = u_pred[k]
        L[k] = float(xk @ Q @ xk + uk @ R @ uk)

    krk = K.T @ R @ K
    Ld = np.array(
        [float(np.trace(Q @ sige_seq[k]) + np.trace(krk @ sige_seq[k])) for k in range(H)],
        dtype=float,
    )
    dcost = float(np.sum(Ld))

    near_origin = np.linalg.norm(x_curr) <= 1e-2

    sig1_vals = []
    for j6 in range(1, Hcbf):
        if near_origin or L[0] <= 0.0:
            sig1_vals.append(0.0)
        else:
            sig1_vals.append(float((L[j6] / L[0]) ** (1.0 / j6)))
    sig1 = float(np.max(sig1_vals)) if sig1_vals else 0.0

    sig2_vals = []
    for j7 in range(Hcbf, H):
        if near_origin or L[Hcbf - 1] <= 0.0:
            sig2_vals.append(0.0)
        else:
            power = 1.0 / (j7 + 1 - Hcbf)
            sig2_vals.append(float((L[j7] / L[Hcbf - 1]) ** power))
    sig2 = float(np.max(sig2_vals)) if sig2_vals else 0.0

    if sig2 <= 1e-12:
        delta = float("inf") if sig1 > 0.0 else 0.0
    else:
        delta = float(max(sig1 / sig2 - 1.0, 0.0))

    n_tail = H - Hcbf - 1
    if n_tail <= 0:
        geom_sum = 0.0
    elif abs(1.0 - sig2) <= 1e-12:
        geom_sum = float(n_tail)
    else:
        geom_sum = float((1.0 - sig2 ** n_tail) / (1.0 - sig2))

    alpha = float(1.0 - (sig1 ** (Hcbf + 1)) * geom_sum)
    return alpha, sig1, sig2, delta, dcost


def main():
    seed = 171
    figs_dir = "simulations/figs/smpc/multiple_horizon"
    os.makedirs(figs_dir, exist_ok=True)

    n = 4
    m = 2
    h = 0.5
    N = 7
    N_tilde = 3
    T = 100
    A = np.array([[1., 0, h, 0],
                  [0, 1., 0, h],
                  [0, 0, 1., 0],
                  [0, 0, 0, 1.]])
    B = np.array([[h*h/2, 0],
                  [0, h*h/2],
                  [h, 0],
                  [0, h]])
    G = np.array([[h, 0],
                  [0, h],
                  [0, 0],
                  [0, 0]])
    C = np.eye(n)
    Q = 10 * np.diag([0.1, 4, 1, 1])
    R = np.eye(m)
    gamma = 0.8
    wmax = 0.005
    wmin = -0.005
    Ccbf1 = np.array([[5./9], [1.], [0], [0]])
    bcbf1 = np.array([[0.5/9]])
    Ccbf2 = np.array([[1.], [-1.], [0], [0]])
    bcbf2 = np.array([[1.6]])
    epsilon = 0.05
    vmax = 2.
    umin = -5
    umax = 5
    Sigma = 1/12 * (2*wmax)**2 * np.eye(m)
    x0 = np.array([[-0.8], [0.6], [-0.45], [0.65]])
    poles = np.array([0.3, 0.4, 0.25, 0.35])
    K = place_poles(A, B, poles)
    K = K.gain_matrix
    Acl = A - B @ K

    Ccbf = np.vstack((Ccbf1.T, Ccbf2.T))
    bcbf = np.vstack((bcbf1, bcbf2))
    N_eff = N - N_tilde
    q_chain = _chain_quantiles(Acl=Acl, Bw=G, C=Ccbf, gamma=gamma, epsilon=epsilon, a=wmax, p=N_eff)
    q_velocity = _velocity_quantiles(Acl=Acl, Bw=G, epsilon=epsilon, a=wmax, p=N_eff)

    def quantile_provider_chain(i, _eps):
        return q_chain[i - 1, :]

    def quantile_provider_velocity(i, _eps):
        return q_velocity[i - 1, :]

    L1 = np.zeros((4, n))
    L1[0, [2, 3]] = [1, 1]
    L1[1, [2, 3]] = [1, -1]
    L1[2, [2, 3]] = [-1, 1]
    L1[3, [2, 3]] = [-1, -1]
    chance_constraint_position = ChanceConstraintNew(
        n=n,
        m=m,
        N=N,
        N_tilde=N_tilde,
        epsilon=epsilon,
        Cprev=(1 - gamma) * Ccbf,
        Ccurr=-Ccbf,
        b=gamma * bcbf,
        quantile_provider=quantile_provider_chain,
        mean_state_indices=None,
    )
    chance_constraint_velocity = ChanceConstraintNew(
        n=n,
        m=m,
        N=N,
        N_tilde=N_tilde,
        epsilon=epsilon,
        Cprev=np.zeros((4, n)),
        Ccurr=L1,
        b=np.ones((4, 1)) * vmax,
        quantile_provider=quantile_provider_velocity,
        mean_state_indices=None,
    )
    input_constraint = InputTighteningConstraint(
        N=N,
        A=A,
        B=B,
        K=K,
        G=G,
        umin=umin,
        umax=umax,
        wmin=wmin,
        wmax=wmax,
    )
    constraints = [
        input_constraint,
        chance_constraint_position,
        chance_constraint_velocity,
    ]

    # --- plant (true system) ---
    plant = LinearPlant(
        A=A, B=B, C=C, N=T,
        Sigma=Sigma, Gamma=None, G=G,
        # Uniform(-sqrt(3), sqrt(3)) has unit variance.
        process_noise_sampler=UniformNoise(-np.sqrt(3.0), np.sqrt(3.0)),
        measurement_noise_sampler=ZeroNoise(),
        seed=seed,
    )

    # --- controller (MPC) ---
    mpc = TightenedTubeSmpc(N=N, N_tilde=N_tilde, A=A, B=B, G=G,
                        Q=Q, R=R, K=K, Sigma=Sigma, epsilon=epsilon,
                        umin=umin, umax=umax, wmin=wmin, wmax=wmax,
                        constraints=constraints,
                        #ancillary_law=ancillary_law,
                        )

    # --- Quadratic Cost ---
    qc = Quadratic(T, Q, R, Q)

    # --- script-level simulation loop with alpha logic from centralopt.m ---
    H = N
    Hcbf = N_eff
    sige_seq = _precompute_error_covariances(Acl=Acl, Bw=G, Sigma_w=Sigma, H=H)

    x = []
    y = []
    u = []
    alpha_hist = []
    sig1_hist = []
    sig2_hist = []
    delta_hist = []
    dcost_hist = []

    mpc.initialize()

    x0_state = plant.set_initial_state(x0)
    x.append(x0_state)
    y0 = plant.measure(x0_state)
    y.append(y0)

    for k in range(T):
        uk = mpc.compute(y[k])
        z_val = mpc.z.value
        if z_val is None:
            raise RuntimeError("Solver returned no primal solution for alpha computation.")
        z_solution = np.asarray(z_val, dtype=float).reshape(-1)

        alpha_k, sig1_k, sig2_k, delta_k, dcost_k = _alpha_metrics_from_solution(
            z_solution=z_solution,
            x_curr=np.asarray(y[k], dtype=float).reshape(-1),
            Q=Q,
            R=R,
            K=K,
            sige_seq=sige_seq,
            H=H,
            Hcbf=Hcbf,
        )
        alpha_hist.append(alpha_k)
        sig1_hist.append(sig1_k)
        sig2_hist.append(sig2_k)
        delta_hist.append(delta_k)
        dcost_hist.append(dcost_k)

        u.append(uk)
        x_next = plant.propagate(x[k], uk)
        x.append(x_next)
        y_next = plant.measure(x_next)
        y.append(y_next)

    cl_cost = qc(x, u)

    # --- plots ---
    plt.figure()
    plt.plot([xi[0, 0] for xi in x])
    plt.title("x[0] (position)")
    plt.savefig(f"{figs_dir}/x0.png")

    plt.figure()
    plt.plot([xi[1, 0] for xi in x])
    plt.title("x[1] (velocity)")
    plt.savefig(f"{figs_dir}/x1.png")

    plt.figure()
    plt.plot([ui[0, 0] for ui in u])
    plt.title("u (control)")
    plt.savefig(f"{figs_dir}/u.png")

    plt.figure()
    plt.plot(alpha_hist)
    plt.title("alpha (centralopt logic)")
    plt.savefig(f"{figs_dir}/alpha.png")

    print(f"Saved plots! cost={cl_cost:.4f}, alpha_min={np.min(alpha_hist):.6f}")


if __name__ == "__main__":
    main()
