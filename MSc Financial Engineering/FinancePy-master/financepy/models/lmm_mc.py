##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################

from enum import Enum
import numpy as np
from numba import jit, njit, float64, int64  # , prange DOES NOT WORK ON GITHUB

from ..utils.error import FinError
from ..utils.math import N
from ..utils.math import norminvcdf
from ..models.sobol import get_uniform_sobol

# TO DO: SHIFTED LOGNORMAL
# TO DO: TERMINAL MEASURE
# TO DO:: CALIBRATION

useParallel = False

###############################################################################

""" This module manages the Ibor Market Model and so stores a specific MC
    forward rate simulation consisting of a 3D matrix of num_paths x numForwards
    x (numForwards-1)/2 elements. This is a lognormal model although a shifted
    Lognormal rate is also allowed. Implementations include 1 factor, M factor
    where the volatility curve per factor is provided and a full N-factor corr-
    elation matrix where a Cholesky is done to decompose the N factors. """

###############################################################################


class ModelLMMModelTypes(Enum):
    LMM_ONE_FACTOR = 1
    LMM_HW_M_FACTOR = 2
    LMM_FULL_N_FACTOR = 3

###############################################################################


def lmm_print_forwards(fwds):
    """ Helper function to display the simulated Ibor rates. """

    num_paths = len(fwds)
    num_times = len(fwds[0])
    numFwds = len(fwds[0][0])

    if num_paths > 10:
        return

    for ip in range(0, num_paths):
        for it in range(0, num_times):

            print("Path: %3d Time: %3d" % (ip, it), end=""),

            for ifwd in range(0, it):
                print("%8s" % ("-"), end=""),

            for ifwd in range(it, numFwds):
                print("%8.4f" % (fwds[ip][it][ifwd]*100.0), end=""),

            print("")


###############################################################################


@njit(float64(int64, int64, float64[:], float64[:], float64[:], float64[:, :]),
      cache=True, fastmath=True, parallel=useParallel)
def lmm_swaption_vol_approx(a, b, fwd0, taus, zetas, rho):
    """ Implements Rebonato's approximation for the swap rate volatility to be
    used when pricing a swaption that expires in period a for a swap maturing
    at the end of period b taking into account the forward volatility term
    structure (zetas) and the forward-forward correlation matrix rho.. """

    num_periods = len(fwd0)

#    if len(taus) != num_periods:
#        raise FinError("Tau vector must have length" + str(num_periods))

#    if len(zetas) != num_periods:
#        raise FinError("Tau vector must have length" + str(num_periods))

#    if len(rho) != num_periods:
#        raise FinError("Rho matrix must have length" + str(num_periods))

#    if len(rho[0]) != num_periods:
#        raise FinError("Rho matrix must have height" + str(num_periods))

    if b > num_periods:
        raise FinError("Swap maturity beyond num_periods.")

    if a == b:
        raise FinError("Swap maturity on swap expiry date")

    p = np.zeros(num_periods)
    p[0] = 1.0 / (1.0 + fwd0[0] * taus[0])
    for ix in range(1, num_periods):
        p[ix] = p[ix-1] / (1.0 + fwd0[ix] * taus[ix])

    wts = np.zeros(num_periods)
    pv01ab = 0.0
    for k in range(a+1, b):
        pv01ab += taus[k] * p[k]

    sab = (p[a] - p[b-1])/pv01ab

    for i in range(a, b):
        wts[i] = taus[i] * p[i] / pv01ab

    swaptionVar = 0.0
    for i in range(a, b):
        for j in range(a, b):
            wti = wts[i]
            wtj = wts[j]
            fi = fwd0[i]
            fj = fwd0[j]
            intsigmaij = 0.0

            for k in range(0, a):
                intsigmaij += zetas[i] * zetas[j] * taus[k]

            term = wti * wtj * fi * fj * rho[i][j] * intsigmaij / (sab**2)
            swaptionVar += term

    taua = 0.0
    for i in range(0, a):
        taua += taus[i]

    taub = 0.0
    for i in range(0, b):
        taub += taus[i]

    swaptionVol = np.sqrt(swaptionVar/taua)
    return swaptionVol


###############################################################################


@njit(float64(int64, int64, float64[:], float64[:, :, :], float64[:]),
      cache=True, fastmath=True, parallel=useParallel)
def lmm_sim_swaption_vol(a, b, fwd0, fwds, taus):
    """ Calculates the swap rate volatility using the forwards generated in the
    simulation to see how it compares to Rebonatto estimate. """

    num_paths = len(fwds)
    numForwards = len(fwds[0])

    if a > numForwards:
        raise FinError("NumPeriods > numForwards")

    if a >= b:
        raise FinError("Swap maturity is before expiry date")

    fwdSwapRateMean = 0.0
    fwdSwapRateVar = 0.0

    for iPath in range(0, num_paths):  # changed from prange

        numeraire = 1.0

        for k in range(0, a):
            numeraire *= (1.0 + taus[k] * fwds[iPath, k, k])

        pv01 = 0.0
        df = 1.0

        for k in range(a, b):
            f = fwds[iPath, a, k]
            tau = taus[k]
            df = df / (1.0 + tau * f)
            pv01 = pv01 + tau * df

        fwdSwapRate = (1.0 - df) / pv01

        fwdSwapRateMean += fwdSwapRate
        fwdSwapRateVar += fwdSwapRate**2

    taua = 0.0
    for i in range(0, a):
        taua += taus[i]

    fwdSwapRateMean /= num_paths
    fwdSwapRateVar = fwdSwapRateVar/num_paths - fwdSwapRateMean**2
    fwdSwapRateVol = np.sqrt(fwdSwapRateVar/taua)
    fwdSwapRateVol /= fwdSwapRateMean
    return fwdSwapRateVol

###############################################################################


@njit(float64[:, :](int64, int64, int64, float64[:, :, :]),
      cache=True, fastmath=True, parallel=useParallel)
def lmm_fwd_fwd_correlation(numForwards, num_paths, iTime, fwds):
    """ Extract forward forward correlation matrix at some future time index
    from the simulated forward rates and return the matrix. """

    size = numForwards - iTime
    fwdCorr = np.zeros((size, size))

    for iFwd in range(iTime, numForwards):
        for jFwd in range(iFwd, numForwards):

            sumfwdi = 0.0
            sumfwdj = 0.0
            sumfwdifwdi = 0.0
            sumfwdifwdj = 0.0
            sumfwdjfwdj = 0.0

            for p in range(0, num_paths):  # changed from prange
                dfwdi = fwds[p, iTime, iFwd] - fwds[p, iTime-1, iFwd]
                dfwdj = fwds[p, iTime, jFwd] - fwds[p, iTime-1, jFwd]
                sumfwdi += dfwdi
                sumfwdj += dfwdj
                sumfwdifwdi += dfwdi * dfwdi
                sumfwdifwdj += dfwdi * dfwdj
                sumfwdjfwdj += dfwdj * dfwdj

            avgfwdi = sumfwdi / num_paths
            avgfwdj = sumfwdj / num_paths
            avgfwdifwdi = sumfwdifwdi / num_paths
            avgfwdifwdj = sumfwdifwdj / num_paths
            avgfwdjfwdj = sumfwdjfwdj / num_paths

            covii = avgfwdifwdi - avgfwdi * avgfwdi
            covjj = avgfwdjfwdj - avgfwdj * avgfwdj
            covij = avgfwdifwdj - avgfwdi * avgfwdj
            corr = covij / np.sqrt(covii*covjj)

            if abs(covii*covjj) > 1e-20:
                fwdCorr[iFwd-iTime][jFwd-iTime] = corr
                fwdCorr[jFwd-iTime][iFwd-iTime] = corr
            else:
                fwdCorr[iFwd-iTime][jFwd-iTime] = 0.0
                fwdCorr[jFwd-iTime][iFwd-iTime] = 0.0

    return fwdCorr

###############################################################################


@njit(float64[:](float64[:], float64[:], int64, float64, float64[:]),
      cache=True, fastmath=True)
def lmm_price_caps_black(fwd0, volCaplet, p, K, taus):
    """ Price a strip of capfloorlets using Black's model using the time grid
    of the LMM model. The prices can be compared with the LMM model prices. """

    caplet = np.zeros(p+1)
    discFwd = np.zeros(p+1)

    if K <= 0.0:
        raise FinError("Negative strike not allowed.")

    # Set up initial term structure
    discFwd[0] = 1.0 / (1.0 + fwd0[0] * taus[0])
    for i in range(1, p):
        discFwd[i] = discFwd[i-1] / (1.0 + fwd0[i] * taus[i])

    # Price ATM caplets
    texp = 0.0

    for i in range(1, p):  # 1 to p-1

        K = fwd0[i]
        texp += taus[i]
        vol = volCaplet[i]
        F = fwd0[i]
        d1 = (np.log(F/K) + vol * vol * texp / 2.0) / vol / np.sqrt(texp)
        d2 = d1 - vol * np.sqrt(texp)
        caplet[i] = (F * N(d1) - K * N(d2)) * taus[i] * discFwd[i]

    return caplet

###############################################################################


@njit(float64[:, :](float64[:, :], int64), cache=True, fastmath=True)
def sub_matrix(t, N):
    """ Returns a submatrix of correlation matrix at later time step in the LMM
    simulation which is then used to generate correlated Gaussian RVs. """

    lent = len(t)
    result = np.zeros((lent-N-1, lent-N-1))

    for i in range(N + 1, lent):
        for j in range(N + 1, lent):
            result[i - N - 1][j - N - 1] = t[i][j]

    return result

###############################################################################


@njit(float64[:, :](float64[:, :]), cache=True, fastmath=True)
def cholesky_np(rho):
    """ Numba-compliant wrapper around Numpy cholesky function. """
    chol = np.linalg.cholesky(rho)
    return chol

###############################################################################


@jit(float64[:, :, :](int64, int64, float64[:], float64[:], float64[:, :],
                      float64[:], int64),
     cache=True, fastmath=True, parallel=useParallel)
def lmm_simulate_fwds_nf(numForwards, num_paths, fwd0, zetas, correl, taus, seed):
    """ Full N-Factor Arbitrage-free simulation of forward Ibor discount in the
    spot measure given an initial forward curve, volatility term structure and
    full rank correlation structure. Cholesky decomposition is used to extract
    the factor weights. The number of forwards at time 0 is given. The 3D
    matrix of forward rates by path, time and forward point is returned.
    WARNING: NEED TO CHECK THAT CORRECT VOLATILITY IS BEING USED (OFF BY ONE
    BUG NEEDS TO BE RULED OUT) """

    np.random.seed(seed)

    # Even number of paths for antithetics
    num_paths = 2 * int(num_paths/2)
    halfNumPaths = int(num_paths/2)

    fwd = np.empty((num_paths, numForwards, numForwards))
    fwdB = np.zeros(numForwards)

    discFwd = np.zeros(numForwards)

    # Set up initial term structure
    discFwd[0] = 1.0 / (1.0 + fwd0[0] * taus[0])
    for ix in range(1, numForwards):
        discFwd[ix] = discFwd[ix-1] / (1.0 + fwd0[ix] * taus[ix])

    corr = [None]  # from 0 to p-1
    factors = [None]  # from 0 to p-1

    for ix in range(1, numForwards):  # from 1 to p-1
        matrix = sub_matrix(correl, ix - 1)
        corr.append(matrix)
        chol = cholesky_np(matrix)
        factors.append(chol)

    ###########################################################################
    # I HAVE PROBLEMS AS THE PARALLELISATION CHANGES THE OUTPUT IF RANDS ARE
    # CALCULATED INSIDE THE MAIN LOOP SO I CALCULATE THEM NOW
    ###########################################################################

    if 1 == 1:
        gMatrix = np.empty((num_paths, numForwards, numForwards))
        for iPath in range(0, halfNumPaths):
            for j in range(1, numForwards):
                for k in range(0, numForwards-j):
                    g = np.random.normal()
                    # ANTITHETICS
                    gMatrix[iPath, j, k] = g
                    gMatrix[iPath + halfNumPaths, j, k] = -g

    avgg = 0.0
    stdg = 0.0

    for iPath in range(0, num_paths):

        # Initial value of forward curve at time 0
        for iFwd in range(0, numForwards):
            fwd[iPath, 0, iFwd] = fwd0[iFwd]

        for j in range(1, numForwards):  # TIME LOOP

            dt = taus[j]
            sqrt_dt = np.sqrt(dt)

            for i in range(j, numForwards):  # FORWARDS LOOP

                zi = zetas[i]

                muA = 0.0
                for k in range(j, i+1):
                    rho = corr[j][k-j, i-j]
                    fk = fwd[iPath, j-1, k]
                    zk = zetas[k]
                    tk = taus[k]
                    muA += zi * fk * tk * zk * rho / (1.0 + fk * tk)

                w = 0.0
                for k in range(0, numForwards-j):
                    f = factors[j][i-j, k]
                    w = w + f * gMatrix[iPath, j, k]

                avgg += w
                stdg += w*w

                fwdB[i] = fwd[iPath, j-1, i] \
                    * np.exp(muA * dt - 0.5 * (zi**2) * dt + zi * w * sqrt_dt)

                muB = 0.0
                for k in range(j, i+1):
                    rho = corr[j][k-j, i-j]
                    fk = fwdB[k]
                    zk = zetas[k]
                    tk = taus[k]
                    muB += zi * fk * tk * zk * rho / (1.0 + fk * tk)

                muAvg = 0.5*(muA + muB)
                x = np.exp(muAvg * dt - 0.5 * (zi**2) * dt + zi * w * sqrt_dt)
                fwd[iPath, j, i] = fwd[iPath, j-1, i] * x

    return fwd

###############################################################################


@njit(float64[:, :, :](int64, int64, int64, float64[:], float64[:], float64[:],
                       int64, int64), cache=True, fastmath=True, parallel=useParallel)
def lmm_simulate_fwds_1f(numForwards, num_paths, numeraireIndex, fwd0, gammas,
                         taus, useSobol, seed):
    """ One factor Arbitrage-free simulation of forward Ibor discount in the
    spot measure following Hull Page 768. Given an initial forward curve,
    volatility term structure. The 3D matrix of forward rates by path, time
    and forward point is returned. This function is kept mainly for its
    simplicity and speed.

    NB: The Gamma volatility has an initial entry of zero. This differs from
    Hull's indexing by one and so is why I do not subtract 1 from the index as
    Hull does in his equation 32.14.

    The Number of Forwards is the number of points on the initial curve to the
    trade maturity date.

    But be careful: a cap that matures in 10 years with quarterly caplets has
    40 forwards BUT the last forward to reset occurs at 9.75 years. You should
    not simulate beyond this time. If you give the model 10 years as in the
    Hull examples, you need to simulate 41 (or in this case 11) forwards as the
    final cap or ratchet has its reset in 10 years. """

    if len(gammas) != numForwards:
        raise FinError("Gamma vector does not have right number of forwards")

    if len(fwd0) != numForwards:
        raise FinError("The length of fwd0 is not equal to numForwards")

    if len(taus) != numForwards:
        raise FinError("The length of Taus is not equal to numForwards")

    np.random.seed(seed)
    # Even number of paths for antithetics
    num_paths = 2 * int(num_paths/2)
    halfNumPaths = int(num_paths/2)
    fwd = np.empty((num_paths, numForwards, numForwards))
    fwdB = np.zeros(numForwards)

    num_times = numForwards

    if useSobol == 1:
        numDimensions = num_times
        rands = get_uniform_sobol(halfNumPaths, numDimensions)
        gMatrix = np.empty((num_paths, num_times))
        for iPath in range(0, halfNumPaths):
            for j in range(0, num_times):
                u = rands[iPath, j]
                g = norminvcdf(u)
                gMatrix[iPath, j] = g
                gMatrix[iPath + halfNumPaths, j] = -g
    elif useSobol == 0:
        gMatrix = np.empty((num_paths, num_times))
        for iPath in range(0, halfNumPaths):
            for j in range(0, num_times):
                g = np.random.normal()
                gMatrix[iPath, j] = g
                gMatrix[iPath + halfNumPaths, j] = -g
    else:
        raise FinError("Use Sobol must be 0 or 1")

    for iPath in range(0, num_paths):  # changed from prange
        # Initial value of forward curve at time 0
        for iFwd in range(0, numForwards):
            fwd[iPath, 0, iFwd] = fwd0[iFwd]

        for j in range(0, numForwards-1):  # TIME LOOP
            dtj = taus[j]
            sqrt_dtj = np.sqrt(dtj)
            w = gMatrix[iPath, j]

            for k in range(j, numForwards):  # FORWARDS LOOP
                zkj = gammas[k-j]
                muA = 0.0

                for i in range(j+1, k+1):
                    fi = fwd[iPath, j, i]
                    zij = gammas[i-j]
                    ti = taus[i]
                    muA += zkj * fi * ti * zij / (1.0 + fi * ti)

                # predictor corrector
                x = np.exp(muA * dtj - 0.5*(zkj**2) * dtj + zkj * w * sqrt_dtj)
                fwdB[k] = fwd[iPath, j, k] * x

                muB = 0.0
                for i in range(j+1, k+1):
                    fi = fwdB[k]
                    zij = gammas[i-j]
                    ti = taus[i]
                    muB += zkj * fi * ti * zij / (1.0 + fi * ti)

                muC = 0.5*(muA+muB)

                x = np.exp(muC*dtj - 0.5 * (zkj**2) * dtj + zkj * w * sqrt_dtj)
                fwd[iPath, j+1, k] = fwd[iPath, j, k] * x

    return fwd

###############################################################################


@jit(float64[:, :, :](int64, int64, int64, int64, float64[:], float64[:, :],
                      float64[:], int64, int64),
     cache=True, fastmath=True, parallel=useParallel)
def lmm_simulate_fwds_mf(numForwards, numFactors, num_paths, numeraireIndex, fwd0,
                         lambdas, taus, useSobol, seed):
    """ Multi-Factor Arbitrage-free simulation of forward Ibor discount in the
    spot measure following Hull Page 768. Given an initial forward curve,
    volatility factor term structure. The 3D matrix of forward rates by path,
    time and forward point is returned. """

    np.random.seed(seed)

    if len(lambdas) != numFactors:
        raise FinError("Lambda does not have the right number of factors")

    if len(lambdas[0]) != numForwards:
        raise FinError("Lambda does not have the right number of forwards")

    # Even number of paths for antithetics
    num_paths = 2 * int(num_paths/2)
    halfNumPaths = int(num_paths/2)
    fwd = np.empty((num_paths, numForwards, numForwards))
    fwdB = np.zeros(numForwards)

    num_times = numForwards

    if useSobol == 1:
        numDimensions = num_times * numFactors
        rands = get_uniform_sobol(halfNumPaths, numDimensions)
        gMatrix = np.empty((num_paths, num_times, numFactors))
        for iPath in range(0, halfNumPaths):
            for j in range(0, num_times):
                for q in range(0, numFactors):
                    col = j*numFactors + q
                    u = rands[iPath, col]
                    g = norminvcdf(u)
                    gMatrix[iPath, j, q] = g
                    gMatrix[iPath + halfNumPaths, j, q] = -g
    elif useSobol == 0:
        gMatrix = np.empty((num_paths, num_times, numFactors))
        for iPath in range(0, halfNumPaths):
            for j in range(0, num_times):
                for q in range(0, numFactors):
                    g = np.random.normal()
                    gMatrix[iPath, j, q] = g
                    gMatrix[iPath + halfNumPaths, j, q] = -g
    else:
        raise FinError("Use Sobol must be 0 or 1.")

    for iPath in range(0, num_paths):
        # Initial value of forward curve at time 0
        for iFwd in range(0, numForwards):
            fwd[iPath, 0, iFwd] = fwd0[iFwd]

        for j in range(0, numForwards-1):  # TIME LOOP
            dtj = taus[j]
            sqrt_dtj = np.sqrt(dtj)

            for k in range(j, numForwards):  # FORWARDS LOOP

                muA = 0.0
                for i in range(j+1, k+1):
                    fi = fwd[iPath, j, i]
                    ti = taus[i]
                    zz = 0.0
                    for q in range(0, numFactors):
                        zij = lambdas[q][i-j]
                        zkj = lambdas[q][k-j]
                        zz += zij * zkj
                    muA += fi * ti * zz / (1.0 + fi * ti)

                itoTerm = 0.0
                for q in range(0, numFactors):
                    itoTerm += lambdas[q][k-j] * lambdas[q][k-j]

                randomTerm = 0.0
                for q in range(0, numFactors):
                    wq = gMatrix[iPath, j, q]
                    randomTerm += lambdas[q][k-j] * wq
                randomTerm *= sqrt_dtj

                x = np.exp(muA * dtj - 0.5 * itoTerm * dtj + randomTerm)
                fwdB[k] = fwd[iPath, j, k] * x

                muB = 0.0
                for i in range(j+1, k+1):
                    fi = fwdB[k]
                    ti = taus[i]
                    zz = 0.0
                    for q in range(0, numFactors):
                        zij = lambdas[q][i-j]
                        zkj = lambdas[q][k-j]
                        zz += zij * zkj
                    muB += fi * ti * zz / (1.0 + fi * ti)

                muC = 0.5 * (muA + muB)

                x = np.exp(muC * dtj - 0.5 * itoTerm * dtj + randomTerm)
                fwd[iPath, j+1, k] = fwd[iPath, j, k] * x

    return fwd

###############################################################################


@njit(float64[:](int64, int64, float64, float64[:], float64[:, :, :],
                 float64[:], int64),
      cache=True, fastmath=True, parallel=useParallel)
def lmm_cap_flr_pricer(numForwards, num_paths, K, fwd0, fwds, taus, isCap):
    """ Function to price a strip of cap or floorlets in accordance with the
    simulated forward curve dynamics. """

    maxPaths = len(fwds)
    maxForwards = len(fwds[0])

    if numForwards > maxForwards:
        raise FinError("NumForwards > maxForwards")

    if num_paths > maxPaths:
        raise FinError("NumPaths > MaxPaths")

    discFactor = np.zeros(numForwards)
    capFlrLets = np.zeros(numForwards-1)
    capFlrLetValues = np.zeros(numForwards-1)
    numeraire = np.zeros(numForwards)

    for iPath in range(0, num_paths):

        periodRoll = 1.0
        libor = fwds[iPath, 0, 0]
        capFlrLets[0] = max(K - libor, 0.0) * taus[0]

        # Now loop over the caplets starting with one that fixes immediately
        # but which may have intrinsic value that cannot be ignored.
        for j in range(0, numForwards):

            libor = fwds[iPath, j, j]
            if j == 1:
                if isCap == 0:
                    capFlrLets[j] = max(K - libor, 0.0) * taus[j]
                else:
                    capFlrLets[j] = max(libor - K, 0.0) * taus[j]

                numeraire[0] = 1.0 / discFactor[0]
            else:
                if isCap == 1:
                    capFlrLets[j] = max(libor - K, 0.0) * taus[j]
                elif isCap == 0:
                    capFlrLets[j] = max(K - libor, 0.0) * taus[j]
                else:
                    raise FinError("isCap should be 0 or 1")

            periodRoll = (1.0 + libor * taus[j])
            numeraire[j] = numeraire[j - 1] * periodRoll

        for iFwd in range(0, numForwards):
            denom = abs(numeraire[iFwd]) + 1e-12
            capFlrLetValues[iFwd] += capFlrLets[iFwd] / denom

    for iFwd in range(0, numForwards):
        capFlrLetValues[iFwd] /= num_paths

    return capFlrLetValues

###############################################################################


@njit(float64(float64, int64, int64, float64[:], float64[:, :, :],
              float64[:]), cache=True, fastmath=True, parallel=useParallel)
def lmm_swap_pricer(cpn, num_periods, num_paths, fwd0, fwds, taus):
    """ Function to reprice a basic swap using the simulated forward Ibors.
    """

    maxPaths = len(fwds)
    maxForwards = len(fwds[0])

    if num_periods > maxForwards:
        raise FinError("NumPeriods > numForwards")

    if num_paths > maxPaths:
        raise FinError("NumPaths > MaxPaths")

    discFactor = np.zeros(maxForwards)
    numeraire = np.zeros(maxForwards)
    sumFixed = 0.0
    sumFloat = 0.0
    fixedFlows = np.zeros(maxForwards)
    floatFlows = np.zeros(maxForwards)

    # Set up initial term structure
    discFactor[0] = 1.0 / (1.0 + fwd0[0] * taus[0])
    for ix in range(1, maxForwards):
        discFactor[ix] = discFactor[ix-1] / (1.0 + fwd0[ix] * taus[ix])

    for iPath in range(0, num_paths):

        periodRoll = 1.0
        libor = fwds[iPath, 0, 0]
        floatFlows[0] = libor * taus[0]
        fixedFlows[0] = cpn * taus[0]
        numeraire[0] = 1.0 / discFactor[0]

        for j in range(1, num_periods):  # TIME LOOP

            libor = fwds[iPath, j, j]

            if j == 1:
                fixedFlows[j] = cpn * taus[j]
                floatFlows[j] = libor * taus[j]
            else:
                fixedFlows[j] = fixedFlows[j-1] * periodRoll + cpn * taus[j]
                floatFlows[j] = floatFlows[j-1] * periodRoll + libor * taus[j]

            periodRoll = (1.0 + libor * taus[j])
            numeraire[j] = numeraire[j - 1] * periodRoll

        for iFwd in range(0, num_periods):
            sumFloat += floatFlows[iFwd] / numeraire[iFwd]
            sumFixed += fixedFlows[iFwd] / numeraire[iFwd]

    sumFloat /= num_paths
    sumFixed /= num_paths
    v = sumFixed - sumFloat
    pv01 = sumFixed/cpn
    swap_rate = sumFloat/pv01

    print("FLOAT LEG:", sumFloat)
    print("FIXED LEG:", sumFixed)
    print("SWAP RATE:", swap_rate)
    print("NET VALUE:", v)
    return v

###############################################################################


@njit(float64(float64, int64, int64, int64, float64[:], float64[:, :, :],
              float64[:], int64), cache=True, fastmath=True, parallel=useParallel)
def lmm_swaption_pricer(strike, a, b, num_paths, fwd0, fwds, taus, isPayer):
    """ Function to price a European swaption using the simulated forward
    discount. """

    maxPaths = len(fwds)
    maxForwards = len(fwds[0])

    if a > maxForwards:
        raise FinError("NumPeriods > numForwards")

    if a >= b:
        raise FinError("Swap maturity is before expiry date")

    if num_paths > maxPaths:
        raise FinError("NumPaths > MaxPaths")

    discFactor = np.zeros(maxForwards)
#    pv01 = np.zeros(maxForwards)

    # Set up initial term structure
    discFactor[0] = 1.0 / (1.0 + fwd0[0] * taus[0])
    for ix in range(1, b):
        discFactor[ix] = discFactor[ix-1] / (1.0 + fwd0[ix] * taus[ix])

    sumPayRecSwaption = 0.0

    for iPath in range(0, num_paths):

        numeraire = 1.0
        for k in range(0, a):
            numeraire *= (1.0 + taus[k] * fwds[iPath, k, k])

        pv01 = 0.0
        df = 1.0

        # Value the swap as if we were at time a with forward curve known
        for k in range(a, b):
            f = fwds[iPath, a, k]
            tau = taus[k]
            df = df / (1.0 + tau * f)
            pv01 = pv01 + tau * df

        fwdSwapRate = (1.0 - df) / pv01

        if isPayer == 1:
            payRecSwaption = max(fwdSwapRate - strike, 0.0) * pv01
        elif isPayer == 0:
            payRecSwaption = max(strike - fwdSwapRate, 0.0) * pv01
        else:
            raise FinError("Unknown payRecSwaption value - must be 0 or 1")

        sumPayRecSwaption += payRecSwaption / (abs(numeraire) + 1e-10)

    payRecPrice = sumPayRecSwaption / num_paths
    return payRecPrice

###############################################################################


@njit(float64[:](float64, int64, int64, float64[:], float64[:, :, :],
                 float64[:]), cache=True, fastmath=True, parallel=useParallel)
def lmm_ratchet_caplet_pricer(spread, num_periods, num_paths, fwd0, fwds, taus):
    """ Price a ratchet using the simulated Ibor rates."""

    maxPaths = len(fwds)
    maxForwards = len(fwds[0][0])

    if num_periods > maxForwards:
        raise FinError("NumPeriods > numForwards")

    if num_paths > maxPaths:
        raise FinError("NumPaths > MaxPaths")

    discFactor = np.zeros(maxForwards)
    numeraire = np.zeros(maxForwards)
    ratchetCaplets = np.zeros(maxForwards)
    ratchetCapletValues = np.zeros(maxForwards)

    # Set up initial term structure
    discFactor[0] = 1.0 / (1.0 + fwd0[0] * taus[0])
    for ix in range(1, maxForwards):
        discFactor[ix] = discFactor[ix-1] / (1.0 + fwd0[ix] * taus[ix])

    for iPath in range(0, num_paths):

        periodRoll = 1.0
        libor = fwds[iPath, 0, 0]
        ratchetCaplets[0] = 0.0

        for j in range(1, num_periods):  # TIME LOOP

            prevIbor = libor
            K = prevIbor + spread
            libor = fwds[iPath, j, j]

            if j == 1:
                ratchetCaplets[j] = max(libor - K, 0.0) * taus[j]
                numeraire[0] = 1.0 / discFactor[0]
            else:
                ratchetCaplets[j] = max(libor - K, 0.0) * taus[j]

            periodRoll = (1.0 + libor * taus[j])
            numeraire[j] = numeraire[j - 1] * periodRoll

        for iFwd in range(0, num_periods):
            ratchetCapletValues[iFwd] += ratchetCaplets[iFwd] / numeraire[iFwd]

    for iFwd in range(0, num_periods):
        ratchetCapletValues[iFwd] /= num_paths

    return ratchetCapletValues

###############################################################################


@njit(float64(int64, float64, int64, int64, float64[:], float64[:, :, :],
              float64[:]), cache=True, fastmath=True, parallel=useParallel)
def lmm_flexi_cap_pricer(maxCaplets, K, num_periods, num_paths, fwd0, fwds, taus):
    """ Price a flexicap using the simulated Ibor rates."""

    maxPaths = len(fwds)
    maxForwards = len(fwds[0][0])

    if num_periods > maxForwards:
        raise FinError("NumPeriods > numForwards")

    if num_paths > maxPaths:
        raise FinError("NumPaths > MaxPaths")

    discFactor = np.zeros(maxForwards)
    numeraire = np.zeros(maxForwards)
    flexiCaplets = np.zeros(maxForwards)
    flexiCapletValues = np.zeros(maxForwards)

    # Set up initial term structure
    discFactor[0] = 1.0 / (1.0 + fwd0[0] * taus[0])
    for ix in range(1, maxForwards):
        discFactor[ix] = discFactor[ix-1] / (1.0 + fwd0[ix] * taus[ix])

    for iPath in range(0, num_paths):

        periodRoll = 1.0
        libor = fwds[iPath, 0, 0]
        flexiCaplets[0] = 0.0

        numCapletsLeft = maxCaplets

        for j in range(1, num_periods):  # TIME LOOP

            libor = fwds[iPath, j, j]

            if j == 1:
                if libor > K and numCapletsLeft > 0:
                    flexiCaplets[j] = max(libor - K, 0.0) * taus[j]
                    numCapletsLeft -= 1
                numeraire[0] = 1.0 / discFactor[0]
            else:
                if libor > K and numCapletsLeft > 0:
                    flexiCaplets[j] = max(libor - K, 0.0) * taus[j]
                    numCapletsLeft -= 1

            periodRoll = (1.0 + libor * taus[j])
            numeraire[j] = numeraire[j - 1] * periodRoll

        for iFwd in range(0, num_periods):
            flexiCapletValues[iFwd] += flexiCaplets[iFwd] / numeraire[iFwd]

    for iFwd in range(0, num_periods):
        flexiCapletValues[iFwd] /= num_paths

    flexiCapValue = 0.0
    for iFwd in range(0, num_periods):
        flexiCapValue += flexiCapletValues[iFwd]

    return flexiCapValue

###############################################################################


@njit(float64[:](float64, int64, int64, float64[:], float64[:, :, :],
                 float64[:]), cache=True, fastmath=True, parallel=useParallel)
def lmm_sticky_caplet_pricer(spread, num_periods, num_paths, fwd0, fwds, taus):
    """ Price a sticky cap using the simulated Ibor rates. """

    maxPaths = len(fwds)
    maxForwards = len(fwds[0][0])

    if num_periods > maxForwards:
        raise FinError("NumPeriods > numForwards")

    if num_paths > maxPaths:
        raise FinError("NumPaths > MaxPaths")

    discFactor = np.zeros(maxForwards)
    numeraire = np.zeros(maxForwards)
    stickyCaplets = np.zeros(maxForwards)
    stickyCapletValues = np.zeros(maxForwards)

    # Set up initial term structure
    discFactor[0] = 1.0 / (1.0 + fwd0[0] * taus[0])
    for ix in range(1, maxForwards):
        discFactor[ix] = discFactor[ix-1] / (1.0 + fwd0[ix] * taus[ix])

    for iPath in range(0, num_paths):

        periodRoll = 1.0
        libor = fwds[iPath, 0, 0]
        stickyCaplets[0] = 0.0
        K = libor

        for j in range(1, num_periods):  # TIME LOOP

            prevIbor = libor
            K = min(prevIbor, K) + spread
            libor = fwds[iPath, j, j]

            if j == 1:
                stickyCaplets[j] = max(libor-K, 0.0) * taus[j]
                numeraire[0] = 1.0 / discFactor[0]
            else:
                stickyCaplets[j] = max(libor - K, 0.0) * taus[j]

            periodRoll = (1.0 + libor * taus[j])
            numeraire[j] = numeraire[j - 1] * periodRoll

        for iFwd in range(0, num_periods):
            stickyCapletValues[iFwd] += stickyCaplets[iFwd] / numeraire[iFwd]

    for iFwd in range(0, num_periods):
        stickyCapletValues[iFwd] /= num_paths

    return stickyCapletValues

###############################################################################
