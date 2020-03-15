# -*- coding: utf-8 -*-
from math import log
import numpy as np
from scipy.optimize import minimize


def _rootsFinder(fun, jac, bounds, npoints, method):
    """
    Find possible roots of a scalar function    
    method : str
        'regular' : regular sample of the search interval,
        'random' : uniform (distribution) sample of the search interval
    
    Return possible roots of the function
    """
    if method == 'regular':
        step = (bounds[1] - bounds[0]) / (npoints + 1)
        X0 = np.arange(bounds[0] + step, bounds[1], step)
    elif method == 'random':
        X0 = np.random.uniform(bounds[0], bounds[1], npoints)

    def objFun(X, f, jac):
        g = 0
        j = np.zeros(X.shape)
        i = 0
        for x in X:
            fx = f(x)
            g = g + fx ** 2
            j[i] = 2 * fx * jac(x)
            i = i + 1
        return g, j

    opt = minimize(lambda X: objFun(X, fun, jac), X0,
                   method='L-BFGS-B',
                   jac=True, bounds=[bounds] * len(X0))

    X = opt.x
    np.round(X, decimals=5)
    return np.unique(X)


def _log_likelihood(Y, gamma, sigma):
    """
    Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)
    Returns log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
    """
    n = Y.size
    if gamma != 0:
        tau = gamma / sigma
        L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
    else:
        L = n * (1 + log(Y.mean()))
    return L


class POT:
    """
    This class allows to run POT algorithm on univariate dataset (upper-bound)
    """

    def __init__(self, q=1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def initialize(self, init_data, level=0.02, min_extrema=False):
        self.init_data = np.array(init_data)
        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(level * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = -1*self.init_data[self.init_data < self.init_threshold] + self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init
        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)
        return self.extreme_quantile

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick
        """
        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = _rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = _rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = _log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = _log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q
        Returns quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold - (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold + sigma * (r)

