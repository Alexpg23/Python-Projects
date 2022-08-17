#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 09:04:06 2021

@author: dominicokane
"""

from numba import njit
import numpy as np

##############################################################################

@njit(fastmath=True, cache=True)
def fit(alpha, x, y):
    ''' Perform a one-factor linear regression on a vector of x and y values '''

    ftol=0.000001
    max_iter=1000

    converged = False
    num_iter = 0
    m = x.shape[0] # number of samples
    y = y.reshape(-1,1) # make y into a column vector
    
    # Initial guesses
    theta0 = 1.0
    theta1 = 1.0
    
    J = 0.0
    for i in range(0, m):
        J = J + (theta0 + theta1 * x[i,0] - y[i,0])**2
    J = J/m/2.0
    
    while not converged:

        # for each training sample, compute the gradient wrt t0 and t1
        s0 = 0.0
        for i in range(0, m):
            s0 = s0 + (theta0 + theta1 * x[i,0] - y[i,0])

        s1 = 0.0
        for i in range(0, m):
            s1 = s1 + (theta0 + theta1 * x[i,0] - y[i,0]) * x[i,0]

        grad0 = 1.0/m * s0 
        grad1 = 1.0/m * s1
        theta0 = theta0 - alpha * grad0
        theta1 = theta1 - alpha * grad1
        
        # mean squared error
        e = 0.0
        for i in range(0,m):
            e = e + (theta0 + theta1*x[i,0] - y[i,0])**2
        e = e/m/2.0
        
        if abs(J-e) <= ftol: 
            converged = True

        J = e   # update error 
        num_iter += 1  # update iter
        
        if num_iter == max_iter: # max iterations exceeded
            print("Max iter reached")
            converged = True # Escape loop
    
    return (theta0, theta1, num_iter, e)

##############################################################################

if __name__ == '__main__':

    np.random.seed(1919)
    m = 1000
    
    X = 2.0 * np.random.rand(m,1)
    y = 4.0 + 3.0 * X + np.random.randn(m) 
    
    print(X.shape)
    print(y.shape)
    
    alpha = 0.1
    
    import time
    
    start = time.time()
    z = fit(alpha, X ,y )
    end = time.time()
    print("Time (s):", end - start)
    print(z)
