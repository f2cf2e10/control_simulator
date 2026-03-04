import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.signal import place_poles

from src.application.ports.outbound.controller_models import AncillaryControlLaw
from src.application.services.simulation_service import SimulationService
from src.infrastructure.adapters.outbound.controllers.mpc.constraint.chance_constraint import ChanceConstraint
from src.infrastructure.adapters.outbound.controllers.mpc.constraint.input_tightening_constraint import InputTighteningConstraint
from src.infrastructure.adapters.outbound.plants.linear_plant import LinearPlant
from src.infrastructure.adapters.outbound.noise_samplers import UniformNoise, ZeroNoise
from src.infrastructure.adapters.outbound.cost import Quadratic
from src.infrastructure.adapters.outbound.controllers.mpc.tightened_smpc import TightenedTubeSmpc


def main():
    seed = 1
    figs_dir = "simulations/figs/smpc/multiple_horizon"

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
    wmax = 0.1
    wmin = -0.1
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
    ancillary_law = AncillaryControlLaw(transform=lambda v0, y_k: v0 - K @ y_k)

    def quantile_provider(i, eps): return st.irwinhall(
        i, loc=i * wmin, scale=(wmax - wmin)
    ).ppf(1.0 - eps)
    Ccbf = np.vstack((Ccbf1.T, Ccbf2.T))
    bcbf = np.array([bcbf1, bcbf2])
    L1 = np.zeros((4, n))
    L1[0, [2, 3]] = [-1, -1]
    L1[1, [2, 3]] = [-1, 1]
    L1[2, [2, 3]] = [1, -1]
    L1[3, [2, 3]] = [1, 1]
    chance_constraint_position = ChanceConstraint(
        n=n,
        m=m,
        N=N,
        N_tilde=N_tilde,
        epsilon=epsilon,
        Cprev=(1 - gamma) * Ccbf,
        Ccurr=-Ccbf,
        b=gamma * bcbf,
        quantile_provider=quantile_provider,
        mean_state_indices=None,
    )
    chance_constraint_velocity = ChanceConstraint(
        n=n,
        m=m,
        N=N,
        N_tilde=N_tilde,
        epsilon=epsilon,
        Cprev=np.zeros((4, n)),
        Ccurr=L1,
        b=np.ones((4, 1)) * vmax,
        quantile_provider=quantile_provider,
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
        process_noise_sampler=UniformNoise(wmin, wmax),
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

    # --- application service (owns the path) ---
    sim = SimulationService(
        plant=plant,
        controller=mpc,
        cost=qc,
        N=T,
        x0=x0,
    )

    result = sim.execute()
    x, y, u = result.x, result.y, result.u

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

    print("Saved plots!")


if __name__ == "__main__":
    main()
