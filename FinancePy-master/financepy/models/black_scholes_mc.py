##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################

# TODO Fix this

import numpy as np

from numba import njit, float64, int64
from ..utils.global_types import OptionTypes
from ..utils.error import FinError
from ..models.sobol import get_gaussian_sobol
from math import exp


###############################################################################

def _value_mc_nonumba_nonumpy(s, t, K, option_type, r, q, v, num_paths, seed, useSobol):
    # SLOWEST - No use of NUMPY vectorisation or NUMBA

    num_paths = int(num_paths)
    np.random.seed(seed)
    mu = r - q
    v2 = v ** 2
    vsqrtt = v * np.sqrt(t)
    payoff = 0.0

    if useSobol == 1:
        g = get_gaussian_sobol(num_paths, 1)[:, 0]
    else:
        g = np.random.standard_normal(num_paths)

    ss = s * exp((mu - v2 / 2.0) * t)

    if option_type == OptionTypes.EUROPEAN_CALL.value:

        for i in range(0, num_paths):
            s_1 = ss * exp(+g[i] * vsqrtt)
            s_2 = ss * exp(-g[i] * vsqrtt)
            payoff += max(s_1 - K, 0.0)
            payoff += max(s_2 - K, 0.0)

    elif option_type == OptionTypes.EUROPEAN_PUT.value:

        for i in range(0, num_paths):
            s_1 = ss * exp(+g[i] * vsqrtt)
            s_2 = ss * exp(-g[i] * vsqrtt)
            payoff += max(K - s_1, 0.0)
            payoff += max(K - s_2, 0.0)

    else:
        raise FinError("Unknown option type.")

    v = payoff * np.exp(-r * t) / num_paths / 2.0
    return v


###############################################################################

def _value_mc_numpy_only(s, t, K, option_type, r, q, v, num_paths, seed, useSobol):
    # Use of NUMPY ONLY

    num_paths = int(num_paths)
    np.random.seed(seed)
    mu = r - q
    v2 = v ** 2
    vsqrtt = v * np.sqrt(t)

    if useSobol == 1:
        g = get_gaussian_sobol(num_paths, 1)[:, 0]
    else:
        g = np.random.standard_normal(num_paths)

    ss = s * np.exp((mu - v2 / 2.0) * t)
    m = np.exp(g * vsqrtt)
    s_1 = ss * m
    s_2 = ss / m

    # Not sure if it is correct to do antithetics with sobols but why not ?
    if option_type == OptionTypes.EUROPEAN_CALL.value:
        payoff_a_1 = np.maximum(s_1 - K, 0.0)
        payoff_a_2 = np.maximum(s_2 - K, 0.0)
    elif option_type == OptionTypes.EUROPEAN_PUT.value:
        payoff_a_1 = np.maximum(K - s_1, 0.0)
        payoff_a_2 = np.maximum(K - s_2, 0.0)
    else:
        raise FinError("Unknown option type.")

    payoff = np.mean(payoff_a_1) + np.mean(payoff_a_2)
    v = payoff * np.exp(-r * t) / 2.0
    return v


###############################################################################

@njit(float64(float64, float64, float64, int64, float64, float64, float64,
              int64, int64, int64), cache=True, fastmath=True)
def _value_mc_numpy_numba(s, t, K, option_type, r, q, v, num_paths, seed, useSobol):
    # Use of NUMPY ONLY

    num_paths = int(num_paths)
    np.random.seed(seed)
    mu = r - q
    v2 = v ** 2
    vsqrtt = v * np.sqrt(t)

    if useSobol == 1:
        g = get_gaussian_sobol(num_paths, 1)[:, 0]
    else:
        g = np.random.standard_normal(num_paths)

    ss = s * np.exp((mu - v2 / 2.0) * t)
    m = np.exp(g * vsqrtt)
    s_1 = ss * m
    s_2 = ss / m

    # Not sure if it is correct to do antithetics with sobols but why not ?
    if option_type == OptionTypes.EUROPEAN_CALL.value:
        payoff_a_1 = np.maximum(s_1 - K, 0.0)
        payoff_a_2 = np.maximum(s_2 - K, 0.0)
    elif option_type == OptionTypes.EUROPEAN_PUT.value:
        payoff_a_1 = np.maximum(K - s_1, 0.0)
        payoff_a_2 = np.maximum(K - s_2, 0.0)
    else:
        raise FinError("Unknown option type.")

    payoff = np.mean(payoff_a_1) + np.mean(payoff_a_2)
    v = payoff * np.exp(-r * t) / 2.0
    return v


###############################################################################

@njit(float64(float64, float64, float64, int64, float64, float64, float64,
              int64, int64, int64), fastmath=True, cache=True)
def _value_mc_numba_only(s, t, K, option_type, r, q, v, num_paths, seed, useSobol):
    # No use of Numpy vectorisation but NUMBA

    num_paths = int(num_paths)
    np.random.seed(seed)
    mu = r - q
    v2 = v ** 2
    vsqrtt = v * np.sqrt(t)
    payoff = 0.0

    if useSobol == 1:
        g = get_gaussian_sobol(num_paths, 1)[:, 0]
    else:
        g = np.random.standard_normal(num_paths)

    ss = s * np.exp((mu - v2 / 2.0) * t)

    if option_type == OptionTypes.EUROPEAN_CALL.value:

        for i in range(0, num_paths):
            gg = g[i]
            s_1 = ss * np.exp(+gg * vsqrtt)
            s_2 = ss * np.exp(-gg * vsqrtt)
            payoff += max(s_1 - K, 0.0)
            payoff += max(s_2 - K, 0.0)

    elif option_type == OptionTypes.EUROPEAN_PUT.value:

        for i in range(0, num_paths):
            gg = g[i]
            s_1 = ss * np.exp(+gg * vsqrtt)
            s_2 = ss * np.exp(-gg * vsqrtt)
            payoff += max(K - s_1, 0.0)
            payoff += max(K - s_2, 0.0)

    else:
        raise FinError("Unknown option type.")

    v = payoff * np.exp(-r * t) / num_paths / 2.0
    return v


###############################################################################


@njit(float64(float64, float64, float64, int64, float64, float64, float64,
              int64, int64, int64), fastmath=True, cache=True, parallel=True)
def _value_mc_numba_parallel(s, t, K, option_type, r, q, v, num_paths, seed, useSobol):
    # No use of Numpy vectorisation but NUMBA

    num_paths = int(num_paths)
    np.random.seed(seed)
    mu = r - q
    v2 = v ** 2
    vsqrtt = v * np.sqrt(t)

    if useSobol == 1:
        g = get_gaussian_sobol(num_paths, 1)[:, 0]
    else:
        g = np.random.standard_normal(num_paths)

    ss = s * np.exp((mu - v2 / 2.0) * t)

    payoff1 = 0.0
    payoff2 = 0.0

    if option_type == OptionTypes.EUROPEAN_CALL.value:

        for i in range(0, num_paths):
            s_1 = ss * exp(+g[i] * vsqrtt)
            s_2 = ss * exp(-g[i] * vsqrtt)
            payoff1 += max(s_1 - K, 0.0)
            payoff2 += max(s_2 - K, 0.0)

    elif option_type == OptionTypes.EUROPEAN_PUT.value:

        for i in range(0, num_paths):
            s_1 = ss * exp(+g[i] * vsqrtt)
            s_2 = ss * exp(-g[i] * vsqrtt)
            payoff1 += max(K - s_1, 0.0)
            payoff2 += max(K - s_2, 0.0)

    else:
        raise FinError("Unknown option type.")

    average_payoff = (payoff1 + payoff2) / 2.0 / num_paths
    v = average_payoff * np.exp(-r * t)
    return v

###############################################################################
