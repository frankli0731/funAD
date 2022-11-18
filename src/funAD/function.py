# -*- coding: utf-8 -*-
"""

This module implements function class, a key component for forward mode autodifferentiation.

"""
from collections.abc import Iterable 
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

    def __init__(self, *fs, x_dim = 1):
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
        self.f = lambda x: [f(x) for f in fs]
        self.x_dim = x_dim

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

        if isinstance(x,Iterable): #input is iterable, has len(.) function
            if len(x) != self.x_dim:
                raise ValueError('Dimension Mismatch')
            else: 
                if len(x) == 1: # 1-d x but pass in a iterable, e.g, if x = [2]
                    x = x[0]
    
        else: # input is NOT iterable, does not have len(.) function
            if self.x_dim != 1:
                raise ValueError('Dimension Mismatch')
        return np.array(self.f(x))

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
        if isinstance(x,Iterable): # input x is iterable, has len() function
            if len(x) != self.x_dim:
                raise ValueError('Dimension Mismatch')
            if self.x_dim == 1:
                x = x[0]
        else: # input x is NOT iterable, does not has len() function
            if self.x_dim != 1:
                raise ValueError('Dimension Mismatch')

        if p is not None: # p provided
            if isinstance(p,Iterable): # input p is iterable, has len() function
                if len(p) != self.x_dim:
                    raise ValueError('Dimension Mismatch')
            if self.x_dim == 1:
                p = p[0]
            else: # input x is NOT iterable, does not has len() function
                if self.x_dim != 1:
                    raise ValueError('Dimension Mismatch')
            return self._grad(x,p)

        else: # p not provided, m-pass, jacobian finding
            J = []
            if self.x_dim == 1:
                return self._grad(x,p)
            else:
                for i in range(self.x_dim):  
                    p = np.identity(self.x_dim)[:,i].tolist()
                    J.append(self._grad(x,p).reshape(-1,1))
            return np.hstack(J)

    def _grad(self,x,p):
        if self.x_dim == 1:
            dual_nums = DualNumber(x,p)
        else: 
            dual_nums = [DualNumber(*input) for input in zip(x,p)]
        result = self.f(dual_nums)
        if isinstance(result,DualNumber):
            return np.array(result.dual)
        else:
            return np.array([d.dual for d in result])
