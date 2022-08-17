##############################################################################
# Copyright (C) 2020 Dominic O'Kane, G Poorna Prudhvi
##############################################################################

import numpy as np
from numba import float64, int64, vectorize, njit

from ..utils.global_types import OptionTypes
from ..utils.global_vars import gSmall
from ..utils.math import n_vect, n_prime_vect
from ..utils.error import FinError
from ..utils.solver_1d import bisection, newton, newton_secant

###############################################################################
# Analytical Black Scholes model implementation and approximations
###############################################################################


@vectorize([float64(float64, float64, float64, float64, float64, float64,
                    int64)], fastmath=True, cache=True)
def bs_value(s, t, k, r, q, v, option_type_value):
    """ Price a derivative using Black-Scholes model. """

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        phi = 1.0
    elif option_type_value == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0
    else:
        raise FinError("Unknown option type value")

    k = np.maximum(k, gSmall)
    t = np.maximum(t, gSmall)
    v = np.maximum(v, gSmall)

    vsqrtT = v * np.sqrt(t)
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / vsqrtT + vsqrtT / 2.0
    d2 = d1 - vsqrtT

    value = phi * ss * n_vect(phi * d1) - phi * kk * n_vect(phi * d2)
    return value

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True)
def bs_delta(s, t, k, r, q, v, option_type_value):
    """ Price a derivative using Black-Scholes model. """

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        phi = +1.0
    elif option_type_value == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0
    else:
        raise FinError("Unknown option type value")

    k = np.maximum(k, gSmall)
    t = np.maximum(t, gSmall)
    v = np.maximum(v, gSmall)

    vsqrtT = v * np.sqrt(t)
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / vsqrtT + vsqrtT / 2.0
    delta = phi * np.exp(-q*t) * n_vect(phi * d1)
    return delta

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True)
def bs_gamma(s, t, k, r, q, v, option_type_value):
    """ Price a derivative using Black-Scholes model. """

    k = np.maximum(k, gSmall)
    t = np.maximum(t, gSmall)
    v = np.maximum(v, gSmall)

    vsqrtT = v * np.sqrt(t)
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / vsqrtT + vsqrtT / 2.0
    gamma = np.exp(-q*t) * n_prime_vect(d1) / s / vsqrtT
    return gamma

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True)
def bs_vega(s, t, k, r, q, v, option_type_value):
    """ Price a derivative using Black-Scholes model. """

    k = np.maximum(k, gSmall)
    t = np.maximum(t, gSmall)
    v = np.maximum(v, gSmall)

    sqrtT = np.sqrt(t)
    vsqrtT = v * sqrtT
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / vsqrtT + vsqrtT / 2.0
    vega = ss * sqrtT * n_prime_vect(d1)
    return vega

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True)
def bs_theta(s, t, k, r, q, v, option_type_value):
    """ Price a derivative using Black-Scholes model. """

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        phi = 1.0
    elif option_type_value == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0
    else:
        raise FinError("Unknown option type value")

    k = np.maximum(k, gSmall)
    t = np.maximum(t, gSmall)
    v = np.maximum(v, gSmall)

    sqrtT = np.sqrt(t)
    vsqrtT = v * sqrtT
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / vsqrtT + vsqrtT / 2.0
    d2 = d1 - vsqrtT
    theta = - ss * n_prime_vect(d1) * v / 2.0 / sqrtT
    theta = theta - phi * r * k * np.exp(-r*t) * n_vect(phi * d2)
    theta = theta + phi * q * ss * n_vect(phi * d1)
    return theta

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True)
def bs_rho(s, t, k, r, q, v, option_type_value):
    """ Price a derivative using Black-Scholes model. """

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        phi = 1.0
    elif option_type_value == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0
    else:
        raise FinError("Unknown option type value")

    k = np.maximum(k, gSmall)
    t = np.maximum(t, gSmall)
    v = np.maximum(v, gSmall)

    sqrtT = np.sqrt(t)
    vsqrtT = v * sqrtT
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / vsqrtT + vsqrtT / 2.0
    d2 = d1 - vsqrtT
    rho = phi * k * t * np.exp(-r*t) * n_vect(phi * d2)
    return rho

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True)
def bs_vanna(s, t, k, r, q, v, option_type_value):
    """ Price a derivative using Black-Scholes model. """

    k = np.maximum(k, gSmall)
    t = np.maximum(t, gSmall)
    v = np.maximum(v, gSmall)

    sqrtT = np.sqrt(t)
    vsqrtT = v * sqrtT
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / vsqrtT + vsqrtT / 2.0
    d2 = d1 - vsqrtT
    vanna = np.exp(-q*t) * sqrtT * n_prime_vect(d1) * (d2/v)
    return vanna

###############################################################################

# @njit(fastmath=True, cache=True)


def _f(sigma, args):

    s = args[0]
    t = args[1]
    k = args[2]
    r = args[3]
    q = args[4]
    price = args[5]
    option_type_value = int(args[6])

    bsPrice = bs_value(s, t, k, r, q, sigma, option_type_value)
    obj = bsPrice - price
    return obj

##############################################################################
# @njit(fastmath=True, cache=True)


def _fvega(sigma, args):

    s = args[0]
    t = args[1]
    k = args[2]
    r = args[3]
    q = args[4]
    option_type_value = int(args[6])
    vega = bs_vega(s, t, k, r, q, sigma, option_type_value)
    return vega

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, int64)], fastmath=True, cache=True)
def bs_intrinsic(s, t, k, r, q, option_type_value):
    """ Calculate the Black-Scholes implied volatility of a European 
    vanilla option using Newton with a fallback to bisection. """

    fwd = s * np.exp((r-q)*t)

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        intrinsic_value = np.exp(-r*t) * max(fwd - k, 0.0)
    else:
        intrinsic_value = np.exp(-r*t) * max(k - fwd, 0.0)

    return intrinsic_value

###############################################################################


# @vectorize([float64(float64, float64, float64, float64, float64, float64,
#                    int64)], fastmath=True, cache=True,  forceobj=True)
def bs_implied_volatility(s, t, k, r, q, price, option_type_value):
    """ Calculate the Black-Scholes implied volatility of a European 
    vanilla option using Newton with a fallback to bisection. """

    fwd = s * np.exp((r-q)*t)

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        intrinsic_value = np.exp(-r*t) * max(fwd - k, 0.0)
    else:
        intrinsic_value = np.exp(-r*t) * max(k - fwd, 0.0)

    divAdjStockPrice = s * np.exp(-q * t)
    df = np.exp(-r * t)

    # Flip ITM call option to be OTM put and vice-versa using put call parity
    if intrinsic_value > 0.0:

        if option_type_value == OptionTypes.EUROPEAN_CALL.value:
            price = price - (divAdjStockPrice - k * df)
            option_type_value = OptionTypes.EUROPEAN_PUT.value
        else:
            price = price + (divAdjStockPrice - k * df)
            option_type_value = OptionTypes.EUROPEAN_CALL.value

        # Update intrinsic based on new option type
        if option_type_value == OptionTypes.EUROPEAN_CALL.value:
            intrinsic_value = np.exp(-r*t) * max(fwd - k, 0.0)
        else:
            intrinsic_value = np.exp(-r*t) * max(k - fwd, 0.0)

    timeValue = price - intrinsic_value

    # Add a tolerance in case it is just numerical imprecision
    if timeValue < 0.0:
        print("Time value", timeValue)
        raise FinError("Option Price is below the intrinsic value")

    ###########################################################################
    # Some approximations which might be used later
    ###########################################################################

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        C = price
    else:
        C = price + (divAdjStockPrice - k * df)

    # Notation in SSRN-id567721.pdf
    X = k * np.exp(-r*t)
    S = s*np.exp(-q*t)
    pi = np.pi

    ###########################################################################
    # Initial point of inflexion
    ###########################################################################

    # arg = np.abs(np.log(fwd/k))
    # sigma0 = np.sqrt(2.0 * arg)

    ###########################################################################
    # Corrado Miller from Hallerbach equation (7)
    ###########################################################################

    cmsigma = 0.0
    # arg = (C - 0.5*(S-X))**2 - ((S-X)**2)/ pi

    # if arg < 0.0:
    #     arg = 0.0

    # cmsigma = (C-0.5*(S-X) + np.sqrt(arg))
    # cmsigma = cmsigma * np.sqrt(2.0*pi) / (S+X)
    # cmsigma = cmsigma / np.sqrt(t)

    ###########################################################################
    # Hallerbach SSRN-id567721.pdf equation (22)
    ###########################################################################

    hsigma = 0.0
    gamma = 2.0
    arg = (2*C+X-S)**2 - gamma * (S+X)*(S-X)*(S-X) / pi / S

    if arg < 0.0:
        arg = 0.0

    hsigma = (2 * C + X - S + np.sqrt(arg))
    hsigma = hsigma * np.sqrt(2.0*pi) / 2.0 / (S+X)
    hsigma = hsigma / np.sqrt(t)

    sigma0 = hsigma

    ###########################################################################

    arglist = [s, t, k, r, q, price, option_type_value]
    argsv = np.array(arglist)

    tol = 1e-6
    sigma = newton(_f, sigma0, _fvega, argsv, tol=tol)

    if sigma is None:
        sigma = bisection(_f, 1e-4, 10.0, argsv, xtol=tol)
        if sigma is None:
            method = "Failed"
        else:
            method = "Bisection"
    else:
        method = "Newton"

    if 1 == 0:
        print("S: %7.2f K: %7.3f T:%5.3f V:%10.7f Sig0: %7.5f CM: %7.5f HL: %7.5f NW: %7.5f %10s" % (
            s, k, t, price, sigma0*100.0, cmsigma*100.0, hsigma*100.0, sigma*100.0, method))

    return sigma

###############################################################################
###############################################################################
# This module contains a number of analytical approximations for the price of
# an American style option starting with Barone-Adesi-Whaley
# https://deriscope.com/docs/Barone_Adesi_Whaley_1987.pdf
###############################################################################
###############################################################################


@njit(fastmath=True, cache=True)
def _fcall(si, *args):
    """ Function to determine ststar for pricing American call options. """

    t = args[0]
    k = args[1]
    r = args[2]
    q = args[3]
    v = args[4]

    b = r - q
    v2 = v*v

    M = 2.0 * r / v2
    W = 2.0 * b / v2
    K = 1.0 - np.exp(-r * t)

    q2 = (1.0 - W + np.sqrt((W - 1.0)**2 + 4.0 * M/K)) / 2.0
    d1 = (np.log(si / k) + (b + v2 / 2.0) * t) / (v * np.sqrt(t))

    obj_fn = si - k
    obj_fn = obj_fn - bs_value(si, t, k, r, q, v, +1)
    obj_fn = obj_fn - (1.0 - np.exp(-q*t) * n_vect(d1)) * si / q2
    return obj_fn

###############################################################################


@njit(fastmath=True, cache=True)
def _fput(si, *args):
    """ Function to determine sstar for pricing American put options. """

    t = args[0]
    k = args[1]
    r = args[2]
    q = args[3]
    v = args[4]

    b = r - q
    v2 = v*v

    W = 2.0 * b / v2
    K = 1.0 - np.exp(-r * t)

    q1 = (1.0 - W - np.sqrt((W - 1.0)**2 + 4.0 * K)) / 2.0
    d1 = (np.log(si / k) + (b + v2 / 2.0) * t) / (v * np.sqrt(t))
    obj_fn = si - k
    obj_fn = obj_fn - bs_value(si, t, k, r, q, v, -1)
    obj_fn = obj_fn - (1.0 - np.exp(-q*t) * n_vect(-d1)) * si / q1
    return obj_fn

###############################################################################
# TODO: NUMBA SPEED UP
###############################################################################


@njit(fastmath=True)
def baw_value(s, t, k, r, q, v, phi):
    """ American Option Pricing Approximation using the Barone-Adesi-Whaley
    approximation for the Black Scholes Model """

    b = r - q

    if phi == 1:

        if b >= r:
            return bs_value(s, t, k, r, q, v, +1)

        argtuple = (t, k, r, q, v)

#        sstar = optimize.newton(_fcall, x0=s, fprime=None, args=argtuple,
#                                tol=1e-7, maxiter=50, fprime2=None)

        sstar = newton_secant(_fcall, x0=s, args=argtuple,
                              tol=1e-7, maxiter=50)

        M = 2.0 * r / (v*v)
        W = 2.0 * b / (v*v)
        K = 1.0 - np.exp(-r * t)
        d1 = (np.log(sstar/k) + (b + v*v / 2.0) * t) / (v * np.sqrt(t))
        q2 = (-1.0 * (W - 1.0) + np.sqrt((W - 1.0)**2 + 4.0 * M/K)) / 2.0
        A2 = (sstar / q2) * (1.0 - np.exp(-q * t) * n_vect(d1))

        if s < sstar:
            return bs_value(s, t, k, r, q, v, +1) + A2 * ((s/sstar)**q2)
        else:
            return s - k

    elif phi == -1:

        argtuple = (t, k, r, q, v)

#        sstar = optimize.newton(_fput, x0=s, fprime=None, args=argtuple,
#                                tol=1e-7, maxiter=50, fprime2=None)

        sstar = newton_secant(_fput, x0=s, args=argtuple,
                              tol=1e-7, maxiter=50)

        v2 = v * v

        M = 2.0 * r / v2
        W = 2.0 * b / v2
        K = 1.0 - np.exp(-r * t)
        d1 = (np.log(sstar / k) + (b + v2 / 2.0) * t) / (v * np.sqrt(t))
        q1 = (-1.0 * (W - 1.0) - np.sqrt((W - 1.0)**2 + 4.0 * M/K)) / 2.0
        a1 = -(sstar / q1) * (1 - np.exp(-q * t) * n_vect(-d1))

        if s > sstar:
            return bs_value(s, t, k, r, q, v, -1) + a1 * ((s/sstar)**q1)
        else:
            return k - s

    else:

        raise FinError("Phi must equal 1 or -1.")

###############################################################################


if __name__ == '__main__':
    # spot_price, strike_price, time_to_expiry, r, b, vol, phi

    # Checking against table 3-1 in Haug
    k = 100.0
    r = 0.10
    q = 0.10

    for t in [0.1, 0.5]:
        for v in [0.15, 0.25, 0.35]:
            for s in [90.0, 100.0, 110.0]:
                bawPrice = baw_value(s, t, k, r, q, v, +1)
                print("%9.5f %9.5f %9.5f %9.5f" % (s, t, v, bawPrice))
