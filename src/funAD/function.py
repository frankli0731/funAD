# -*- coding: utf-8 -*-
"""
This module implements function class, a key component for forward mode autodifferentiation.

"""
import numpy as np
from .dual_number import DualNumber
#from .optimizers import * if need to realize min/max feature in function interface

class function:
    '''
    Create a function object to handle evalutation of a function at particular x coordinates
    and compute corresponding Jacobian through forward mode automatic differentiation.

    '''

    def __init__(self, *f, x_dim=1):
        """
        Record user defined function.
		
        ----------
        f : user defined function(s)
            A function or multiple functions that takes in a vector of x or one x as a scalar and compute arithmetic result.

        """
        self.x_dim = x_dim
        self.function_list = f #list of functions each returning one output

    def _plugin(self,x): # x is an iterable
        results = []
        for f in self.function_list:
            try:
                result = f(x)
            except TypeError:
                try:
                    result = f(*x)
                except TypeError:
                    raise TypeError ('Function fails to process given input.')
            results.append(result)
        return results

    def __call__(self,*x):
        """
        Execute user defined function.
		
        Parameters
        ----------
        f : user defined function
            A function that takes in a vector or scalar of x and compute arithmetic result.
        x : array_like
            An array of numeric values (int or float).
            It could be a scalar (int or float).
			
        Returns
        -------
        f : user defined function
            A function evaluated at x. 
			
        """
        # Preprocess x
        x = np.array(x).reshape(-1) #accepting all kinds of input and make them into 1-d array
        if len(x) != self.x_dim:
            raise ValueError('Dimension Mismatch')
        return np.array(self._plugin(x))

    def grad(self,*x):
        """
        Compute Jacobian based on user specified function(s), based on input independent variable values.
		
        Parameters
        ----------
        x : array_like
            An array of numeric values (int or float).
            It could be a scalar (int or float).
			
        Returns
        -------
        J : array_like
            Jacobian matrix for given function.

        """
        # Preprocess x
        x = np.array(x).reshape(-1) #accepting all kinds of input and make them into 1-d array
        if len(x) != self.x_dim:
            raise ValueError('Dimension Mismatch')
        
        J = []
        for i in range(self.x_dim):  
            p = np.identity(self.x_dim)[:,i].tolist()
            J.append(self._grad(x,p))
        result = np.array(J).T #n x m result

        #putting into desired dimension
        if result.shape[0] == 1:
            result = result[0,:]
        if result.shape[0] == 1:
            result = result[:,0]
        return result

    def _grad(self,x,p):
        """
        Computes the derivative w.r.t. each input.
		
        Parameters
        ----------
        x : array_like
            An array of numeric values (int or float) as independent variables.
            It could be a scalar (int or float).
	p : array_like
            An array of numeric values (int or float) as seed directional vector.
            It could be a scalar (int or float).
			
        Returns
        -------
        J : array_like
            Jacobian matrix for given function.

        """
        dual_nums = [DualNumber(*input) for input in zip(x,p)]
        results = self._plugin(dual_nums)
        return np.array([result.dual for result in results])
    
    # def min(self, x0 = None, optimizer='gd', neg = False, **kwargs):
    #     if optimizer.lower() == 'gd':
    #         gd = GD(**kwargs)
    #         x_star,f_min = gd(self,x0)
    #     elif optimizer.lower() == 'adam':
    #         adam = Adam(**kwargs)
    #             x_star,f_min = adam(f,x0)
    #     else:
    #         raise ValueError('Unidentified optimizer')
    #     return x,f_min
        
