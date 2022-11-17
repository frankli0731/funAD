# -*- coding: utf-8 -*-
"""

This module implements function class, a key component for forward mode autodifferentiation.

"""

import numpy as np
from .dual_number import DualNumber

class function:
    '''
    Create a function object to handle evalutation of a function at particular x coordinates
    and compute corresponding Jacobian through forward mode automatic differentiation.

    Examples
    --------
    Please insert test case

    >>> Please insert test case
    >>> Please insert test case
    Please insert test case   

    '''

    def __init__(self, f=None):
        """
        Record user defined function.

        Note
        ----
        Please add

        Parameters
        ----------
        f : user defined function
            A function that takes in a vector or scalar of x and compute arithmetic result.

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case   

        """
        self.f = f

    def __call__(self,x):
        """
        Execute user defined function.

        Parameters
        ----------
        f : user defined function
            A function that takes in a vector or scalar of x and compute arithmetic result.
        x : array_like
            An array of numeric values (int, float or DualNumber instance).
            It could be a scalar (int or float).

        Returns
        -------
        f : user defined function
            A function evaluated at x. 

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case   

        """
        return self.f(x)

    def grad(self,x,p=None):
        """
        Compute Jacobian based on user specified function(s), x coordinate(s) and seed vector direction(s)

        Note
        ----
        Please add, our p is not properly implemented here

        Parameters
        ----------
        x : array_like
            An array of numeric values (int, float or DualNumbers).
            It could be a scalar (int or float).
        p : array_like
            A seed vector user specified to compute particular directional derivatives.

        Returns
        -------
        J : array_like
            Jacobian matrix for given function.

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case   

        """
        J = []
        for i in range(len(x)): #m-pass
            dual_nums=[]
            p = np.identity(len(x))[:,i].tolist()
            for input in zip(x,p):
                dual_nums.append(DualNumber(*input))
            result = self.f(dual_nums)
            if isinstance(result,DualNumber):
                J.append(result.dual)
                #val = result.real
            else:
                J.append([d.dual for d in result])
                #val=[d.real for d in result]
        J = np.array(J).T
        return J