import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import place_poles

from src.infrastructure.adapters.outbound.controllers.mpc.constraint.chance_constraint import ChanceConstraintNew
from src.infrastructure.adapters.outbound.controllers.mpc.constraint.input_tightening_constraint import InputTighteningConstraint
from src.infrastructure.adapters.outbound.controllers.mpc.constraint.uniform_sum_quantile import (
    quantile_sum_uniform_symmetric,
)
from src.infrastructure.adapters.outbound.plants.linear_plant import LinearPlant
from src.infrastructure.adapters.outbound.noise_samplers import UniformNoise, ZeroNoise
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
                    coeffs.extend(
                        (c @ a_pow @ g_mat @ Bw).reshape(-1).tolist())
            coeffs.extend((-(c @ Bw)).reshape(-1).tolist())
            q[i - 1, j] = quantile_sum_uniform_symmetric(
                1.0 - epsilon, a, np.asarray(coeffs))
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
            q[i - 1, face] = quantile_sum_uniform_symmetric(
                1.0 - eps_face, a, np.asarray(coeffs))
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
        [float(np.trace(Q @ sige_seq[k]) + np.trace(krk @ sige_seq[k]))
         for k in range(H)],
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
    figs_dir = "simulations/figs/smpc/multiple_horizon"
    os.makedirs(figs_dir, exist_ok=True)

    n = 4
    m = 2
    h = 0.5
    N = 11
    T = 100
    Npaths = 100
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
    wmax = 0.01
    wmin = -wmax
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

    seed = 171
    N_tildes = [6, 5, 4, 3, 2]
    Ntrials = 500
    trial_seeds = [seed + i for i in range(Ntrials)]
    sige_seq = _precompute_error_covariances(Acl=Acl, Bw=G, Sigma_w=Sigma, H=N)

    L1 = np.zeros((4, n))
    L1[0, [2, 3]] = [1, 1]
    L1[1, [2, 3]] = [1, -1]
    L1[2, [2, 3]] = [-1, 1]
    L1[3, [2, 3]] = [-1, -1]

    seed_metrics: dict[int, dict[int, dict[str, float | int]]] = {
        trial_seed: {} for trial_seed in trial_seeds
    }

    for N_tilde in N_tildes:
        N_eff = N - N_tilde
        q_chain = _chain_quantiles(
            Acl=Acl, Bw=G, C=Ccbf, gamma=gamma, epsilon=epsilon, a=wmax, p=N_eff
        )
        q_velocity = _velocity_quantiles(
            Acl=Acl, Bw=G, epsilon=epsilon, a=wmax, p=N_eff
        )

        def quantile_provider_chain(i, _eps, q=q_chain):
            return q[i - 1, :]

        def quantile_provider_velocity(i, _eps, q=q_velocity):
            return q[i - 1, :]

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
        mpc = TightenedTubeSmpc(
            N=N,
            N_tilde=N_tilde,
            A=A,
            B=B,
            G=G,
            Q=Q,
            R=R,
            K=K,
            Sigma=Sigma,
            epsilon=epsilon,
            umin=umin,
            umax=umax,
            wmin=wmin,
            wmax=wmax,
            constraints=constraints,
        )

        for trial_seed in trial_seeds:
            print(f"{N_tilde}/{trial_seed}")
            plant = LinearPlant(
                A=A,
                B=B,
                C=C,
                N=T,
                Sigma=Sigma,
                Gamma=None,
                G=G,
                process_noise_sampler=UniformNoise(
                    -np.sqrt(3.0), np.sqrt(3.0)),
                measurement_noise_sampler=ZeroNoise(),
                seed=trial_seed,
            )

            mpc.initialize()
            alpha_hist = []
            x_k = plant.set_initial_state(x0)
            y_k = plant.measure(x_k)

            for _ in range(T):
                u_k = mpc.compute(y_k)
                z_val = mpc.z.value
                if z_val is None:
                    raise RuntimeError(
                        "Solver returned no primal solution for alpha computation."
                    )
                z_solution = np.asarray(z_val, dtype=float).reshape(-1)

                alpha_k, _, _, _, _ = _alpha_metrics_from_solution(
                    z_solution=z_solution,
                    x_curr=np.asarray(y_k, dtype=float).reshape(-1),
                    Q=Q,
                    R=R,
                    K=K,
                    sige_seq=sige_seq,
                    H=N,
                    Hcbf=N_eff,
                )
                alpha_hist.append(alpha_k)

                x_k = plant.propagate(x_k, u_k)
                y_k = plant.measure(x_k)

            alpha_min = float(np.min(alpha_hist))
            avg_exec_time = float(mpc.avg_solve_time)
            metrics: dict[str, float | int] = {
                "seed": trial_seed,
                "alpha_min": alpha_min,
                "avg_execution_time": avg_exec_time,
            }
            seed_metrics[trial_seed][N_tilde] = metrics

    valid_seeds = [
        trial_seed
        for trial_seed in trial_seeds
        if all(
            float(seed_metrics[trial_seed][N_tilde]["alpha_min"]) > 0.0
            for N_tilde in N_tildes
        )
    ]
    selected_seeds = valid_seeds[:Npaths]
    print(f"Valid seeds across all N_tildes: {len(valid_seeds)} / {Ntrials}")
    if len(selected_seeds) < Npaths:
        print(
            f"Only {len(selected_seeds)} valid paths found; requested {Npaths}.")

    per_nt_alpha_min_valid: dict[int, list[float]] = {
        N_tilde: [] for N_tilde in N_tildes}
    simulation_results: list[dict[str, float | int]] = []
    for path_id, trial_seed in enumerate(selected_seeds, start=1):
        valid_row: dict[str, float | int] = {
            "path_id": path_id, "seed": trial_seed}
        for N_tilde in N_tildes:
            metrics = seed_metrics[trial_seed][N_tilde]
            per_nt_alpha_min_valid[N_tilde].append(float(metrics["alpha_min"]))
            valid_row[f"alpha_min_Ntilde_{N_tilde}"] = metrics["alpha_min"]
            valid_row[f"avg_execution_time_Ntilde_{N_tilde}"] = metrics["avg_execution_time"]
        simulation_results.append(valid_row)

    for N_tilde in N_tildes:
        filtered_rows = [seed_metrics[trial_seed][N_tilde]
                         for trial_seed in selected_seeds]
        df_nt = pd.DataFrame(filtered_rows, columns=[
                             "seed", "alpha_min", "avg_execution_time"])
        df_nt.to_csv(f"{figs_dir}/N_tilde_{N_tilde}.csv", index=False)

    pd.DataFrame(simulation_results).to_csv(
        f"{figs_dir}/valid_paths_summary.csv", index=False)

    plt.figure()
    for N_tilde in N_tildes:
        plt.plot(per_nt_alpha_min_valid[N_tilde], label=f"N_tilde={N_tilde}")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("alpha min on valid paths")
    if selected_seeds:
        plt.legend()
    plt.savefig(f"{figs_dir}/alpha.png")


if __name__ == "__main__":
    main()
