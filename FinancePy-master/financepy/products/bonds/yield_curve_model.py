##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################


import numpy as np
from scipy.interpolate import splev
from ...utils.helpers import label_to_string

###############################################################################


class FinCurveFitMethod():
    pass

###############################################################################


class CurveFitPolynomial():

    def __init__(self, power=3):
        self._parentType = FinCurveFitMethod
        self._power = power
        self._coeffs = []

    def _interpolated_yield(self, t):
        yld = np.polyval(self._coeffs, t)
        return yld

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("Power", self._power)

        for c in self._coeffs:
            s += label_to_string("Coefficient", c)

        return s

    def _print(self):
        """ Simple print function for backward compatibility. """
        print(self)

###############################################################################


class CurveFitNelsonSiegel():

    def __init__(self, tau=None, bounds=[(-1, -1, -1, 0.5), (1, 1, 1, 100)]):
        self._parentType = FinCurveFitMethod
        self._beta1 = None
        self._beta2 = None
        self._beta3 = None
        self._tau = tau
        """ Fairly permissive bounds. Only tau1 is 1-100 """
        self._bounds = bounds

    def _interpolated_yield(self, t, beta1=None, beta2=None,
                            beta3=None, tau=None):

        t = np.maximum(t, 1e-10)

        if beta1 is None:
            beta1 = self._beta1

        if beta2 is None:
            beta2 = self._beta2

        if beta3 is None:
            beta3 = self._beta3

        if tau is None:
            tau = self._tau

        theta = t / tau
        expTerm = np.exp(-theta)
        yld = beta1
        yld += beta2 * (1.0 - expTerm) / theta
        yld += beta3 * ((1.0 - expTerm) / theta - expTerm)
        return yld

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("Beta1", self._beta1)
        s += label_to_string("Beta2", self._beta2)
        s += label_to_string("Beta3", self._beta3)
        s += label_to_string("Tau", self._tau)
        return s

    def _print(self):
        """ Simple print function for backward compatibility. """
        print(self)

###############################################################################


class CurveFitNelsonSiegelSvensson():

    def __init__(self, tau1=None, tau2=None,
                 bounds=[(0, -1, -1, -1, 0, 1), (1, 1, 1, 1, 10, 100)]):
        """ Create object to store calibration and functional form of NSS
        parametric fit. """

        self._parentType = FinCurveFitMethod
        self._beta1 = None
        self._beta2 = None
        self._beta3 = None
        self._beta4 = None
        self._tau1 = tau1
        self._tau2 = tau2

        """ I impose some bounds to help ensure a sensible result if
        the user does not provide any bounds. Especially for tau2. """
        self._bounds = bounds

    def _interpolated_yield(self, t, beta1=None, beta2=None, beta3=None,
                            beta4=None, tau1=None, tau2=None):

        # Careful if we get a time zero point
        t = np.maximum(t, 1e-10)

        if beta1 is None:
            beta1 = self._beta1

        if beta2 is None:
            beta2 = self._beta2

        if beta3 is None:
            beta3 = self._beta3

        if beta4 is None:
            beta4 = self._beta4

        if tau1 is None:
            tau1 = self._tau1

        if tau2 is None:
            tau2 = self._tau2

        theta1 = t / tau1
        theta2 = t / tau2
        expTerm1 = np.exp(-theta1)
        expTerm2 = np.exp(-theta2)
        yld = beta1
        yld += beta2 * (1.0 - expTerm1) / theta1
        yld += beta3 * ((1.0 - expTerm1) / theta1 - expTerm1)
        yld += beta4 * ((1.0 - expTerm2) / theta2 - expTerm2)
        return yld

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("Beta1", self._beta1)
        s += label_to_string("Beta2", self._beta2)
        s += label_to_string("Beta3", self._beta3)
        s += label_to_string("Beta4", self._beta3)
        s += label_to_string("Tau1", self._tau1)
        s += label_to_string("Tau2", self._tau2)
        return s

    def _print(self):
        """ Simple print function for backward compatibility. """
        print(self)

###############################################################################


class CurveFitBSpline():

    def __init__(self, power=3, knots=[1, 3, 5, 10]):
        self._parentType = FinCurveFitMethod
        self._power = power
        self._knots = knots
        self._spline = None

    def _interpolated_yield(self, t):
        t = np.maximum(t, 1e-10)
        yld = splev(t, self._spline)
        return yld

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("Power", self._power)
        s += label_to_string("Knots", self._knots)
        s += label_to_string("Spline", self._spline)
        return s

    def _print(self):
        """ Simple print function for backward compatibility. """
        print(self)

###############################################################################
