import numpy as np

from src.infrastructure.adapters.outbound.controllers.nominal_mpc import NominalMpc


def test_nominal_mpc_respects_state_and_input_constraints_horizon_4():
    A = np.array([[1.0, 0.1],
                  [0.0, 1.0]])
    B = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    Q = np.eye(2)
    R = 0.1 * np.eye(2)

    N = 4
    x_min = np.array([-1.0, -1.0])
    x_max = np.array([1.0, 1.0])
    u_min = np.array([-0.2, -0.2])
    u_max = np.array([0.2, 0.2])
    x0 = np.array([[0.8], [-0.7]])

    mpc = NominalMpc(
        N=N,
        A=A,
        B=B,
        Q=Q,
        R=R,
        x_min=x_min,
        x_max=x_max,
        u_min=u_min,
        u_max=u_max,
    )

    _ = mpc.compute(x0)

    z = mpc.z.value
    assert z is not None

    n = mpc.n
    m = mpc.m
    nx = N * n

    x_pred = z[:nx].reshape(N, n)
    u_pred = z[nx:].reshape(N, m)

    tol = 1e-6
    assert np.all(x_pred <= x_max + tol)
    assert np.all(x_pred >= x_min - tol)
    assert np.all(u_pred <= u_max + tol)
    assert np.all(u_pred >= u_min - tol)

    xk = x0.reshape(-1)
    for k in range(N):
        x_next_expected = A @ xk + B @ u_pred[k]
        np.testing.assert_allclose(x_pred[k], x_next_expected, atol=1e-6, rtol=0.0)
        xk = x_pred[k]


def test_nominal_mpc_equality_matrices_match_expected_block_structure():
    N = 4
    n = 2
    m = 2

    A_list = [
        np.array([[1.0, 0.1], [0.0, 1.0]]),
        np.array([[1.1, 0.0], [0.0, 0.9]]),
        np.array([[0.9, 0.2], [0.1, 1.0]]),
        np.array([[1.0, -0.1], [0.0, 1.2]]),
    ]
    B_list = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.5, 0.1], [0.0, 0.8]]),
        np.array([[1.2, 0.0], [0.3, 1.1]]),
        np.array([[0.7, -0.2], [0.0, 0.6]]),
    ]

    mpc = NominalMpc(
        N=N,
        A=A_list,
        B=B_list,
        Q=np.eye(n),
        R=np.eye(m),
    )

    # First constraint is Aeq @ z == E @ x0_param.
    eq_con = mpc.problem_sparse.constraints[0]
    lhs = eq_con.args[0]
    rhs = eq_con.args[1]

    nx = N * n
    nz = nx + N * m

    # Recover Aeq columns by probing lhs with z basis vectors (x0=0).
    Aeq_from_problem = np.zeros((nx, nz))
    mpc.x0_param.value = np.zeros(n)
    for j in range(nz):
        z_basis = np.zeros(nz)
        z_basis[j] = 1.0
        mpc.z.value = z_basis
        Aeq_from_problem[:, j] = np.asarray(lhs.value).reshape(-1)

    # Recover E columns by probing rhs with x0 basis vectors (z=0).
    E_from_problem = np.zeros((nx, n))
    mpc.z.value = np.zeros(nz)
    for j in range(n):
        x0_basis = np.zeros(n)
        x0_basis[j] = 1.0
        mpc.x0_param.value = x0_basis
        E_from_problem[:, j] = np.asarray(rhs.value).reshape(-1)

    # Build expected matrices exactly as in nominal_mpc.py lines 126/129/132.
    Aeq_expected = np.zeros((nx, nz))
    E_expected = np.zeros((nx, n))
    for k in range(N):
        r = k * n
        Aeq_expected[r:r + n, k * n:(k + 1) * n] = np.eye(n)
        if k == 0:
            E_expected[r:r + n, :] = A_list[k]
        else:
            Aeq_expected[r:r + n, (k - 1) * n:k * n] = -A_list[k]
        Aeq_expected[r:r + n, nx + k * m:nx + (k + 1) * m] = -B_list[k]

    np.testing.assert_allclose(Aeq_from_problem, Aeq_expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(E_from_problem, E_expected, atol=1e-12, rtol=0.0)
