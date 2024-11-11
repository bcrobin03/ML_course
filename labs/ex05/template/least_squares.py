# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************

    #we have to solve the equation X.T @ X @ w = X @ y for w
    #which should be w = (X.T @ X)-¹ @ X @ y
    # we use a linear solver to solve this equation Ax = b 
    #where A = X.T @ X and b = X.T @ y
    N = y.shape[0]
    D = tx.shape[1]
    txt = tx.T
    xTx = txt @ tx

    b = txt @ y
    b = b.reshape((D,1))
    
    w = np.linalg.solve(xTx, b)

    #computes the mean square error
    mse = 1/N * np.sum((y - tx@w)**2)

    return w, mse

