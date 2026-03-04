import numpy as np
import scipy.stats as st

from src.infrastructure.adapters.outbound.controllers.mpc.constraint.uniform_sum_quantile import (
    cdf_sum_uniform_symmetric,
    quantile_sum_uniform_symmetric,
)


def test_quantile_matches_irwinhall_for_equal_coefficients():
    # For coeffs = ones(n), sum(coeff_i * U_i), U_i ~ Unif[-a, a],
    # is an affine transform of Irwin-Hall(n).
    a = 0.1
    n = 4
    coeffs = np.ones(n)
    probs = [0.1, 0.5, 0.9]

    for p in probs:
        q = quantile_sum_uniform_symmetric(p, a, coeffs, tol=1e-9)
        q_expected = (2.0 * a) * st.irwinhall(n).ppf(p) - (n * a)
        np.testing.assert_allclose(q, q_expected, rtol=0.0, atol=2e-3)


def test_quantile_matches_empirical_weighted_sum():
    rng = np.random.default_rng(7)
    a = 0.15
    coeffs = np.array([1.0, 0.7, 1.8, 0.5])
    p = 0.95

    q = quantile_sum_uniform_symmetric(p, a, coeffs, tol=1e-8)

    n_samples = 120_000
    u = rng.uniform(-a, a, size=(n_samples, coeffs.size))
    s = u @ coeffs
    empirical_p = np.mean(s <= q)

    assert abs(empirical_p - p) < 0.02
    # CDF should be consistent with quantile as well.
    cdf_at_q = cdf_sum_uniform_symmetric(q, a * np.abs(coeffs))
    assert abs(cdf_at_q - p) < 0.01
