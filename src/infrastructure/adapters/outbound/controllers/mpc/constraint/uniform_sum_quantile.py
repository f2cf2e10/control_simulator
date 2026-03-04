from __future__ import annotations

import warnings
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq


def _cf_sum_uniform_symmetric(t: float, a_terms: np.ndarray) -> float:
    # np.sinc(x) = sin(pi*x)/(pi*x), hence sinc_non_normalized(y) = np.sinc(y/pi).
    return float(np.prod(np.sinc((a_terms * t) / np.pi)))


def cdf_sum_uniform_symmetric(x: float, a_terms: np.ndarray) -> float:
    """
    CDF of S = sum_i V_i, with independent V_i ~ Unif[-a_i, a_i].
    """
    a_terms = np.asarray(a_terms, dtype=float).reshape(-1)
    a_terms = a_terms[a_terms > 0]
    if a_terms.size == 0:
        return 1.0 if x >= 0 else 0.0

    support = float(np.sum(a_terms))
    if x <= -support:
        return 0.0
    if x >= support:
        return 1.0
    if abs(x) < 1e-12:
        return 0.5

    # Use oscillatory quadrature on a regularized integrand:
    # F(x) = 1/2 + (1/pi) * [int sin(tx)*(phi(t)-1)/t dt + int sin(tx)/t dt]
    # and int_0^inf sin(tx)/t dt = (pi/2) * sign(x).
    ax = abs(x)

    def regularized(t: float) -> float:
        if t < 1e-12:
            return 0.0
        return (_cf_sum_uniform_symmetric(t, a_terms) - 1.0) / t

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", IntegrationWarning)
        i_reg, _ = quad(
            regularized,
            0.0,
            np.inf,
            weight="sin",
            wvar=ax,
            limit=400,
            limlst=800,
            epsabs=1e-8,
            epsrel=1e-7,
        )

    if any(issubclass(w.category, IntegrationWarning) for w in wlist):
        # Fallback to finite-interval integration if oscillatory infinite integral
        # does not meet tolerance on this call.
        t_max = 2000.0 / max(float(np.max(a_terms)), 1e-12)

        def direct_regularized(t: float) -> float:
            if t < 1e-12:
                return 0.0
            return np.sin(ax * t) * (_cf_sum_uniform_symmetric(t, a_terms) - 1.0) / t

        i_reg, _ = quad(
            direct_regularized,
            0.0,
            t_max,
            limit=800,
            epsabs=1e-8,
            epsrel=1e-7,
        )

    if x > 0.0:
        cdf = 1.0 + i_reg / np.pi
    else:
        cdf = -i_reg / np.pi
    return float(np.clip(cdf, 0.0, 1.0))


def quantile_sum_uniform_symmetric(
    prob: float,
    a: float,
    coeffs: np.ndarray,
    tol: float = 1e-10,
    maxit: int = 200,
) -> float:
    """
    Quantile of S = sum_i coeff_i * U_i, with independent U_i ~ Unif[-a, a].
    """
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    coeffs = np.abs(coeffs[np.abs(coeffs) > 0])
    if coeffs.size == 0:
        return 0.0

    a_terms = a * coeffs
    support = float(np.sum(a_terms))
    if support == 0.0:
        return 0.0

    if prob <= 0.0:
        return -support
    if prob >= 1.0:
        return support

    def objective(x: float) -> float:
        return cdf_sum_uniform_symmetric(x, a_terms) - prob

    return float(brentq(objective, -support, support, xtol=tol, rtol=tol, maxiter=maxit))
