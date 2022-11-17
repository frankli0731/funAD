# -*- coding: utf-8 -*-
"""

This module implements DualNumber class, a key component 
for forward mode automatic differentiation.

"""
import numpy as np

class DualNumber(object):
    '''
    Create a dual number

    Attributes
    ----------
    _supported_scalars : tuple
        Description of supported numerical types.
    
    '''

    _supported_scalars = (int, float)
    
    def __init__(self,real,dual = None):
        """
        Specify the real and dual part of a dual number

        Note
        ----
        Dual part is typically only set to 1 through external function
        when finding derivative with respect to that dual number.

        Parameters
        ----------
        real : float
            Real part of dual number.
        dual : float
            Dual part of dual number, default to 1 if not specified.

        """
        self.real = real
        if dual == None:
            self.dual = 1.0
        else:
            self.dual = dual

    def __repr__(self):
        """
        Nice string representation of dual number.

        Returns
        ----------
        str

        Examples
        --------
        Print DualNumber

        >>> x = DualNumber(1.0)
        >>> x
        DualNumber(real=1.0, dual=None)

        """
        args = f"real={repr(self.real)}, dual={repr(self.dual)}"
        return f"{type(self).__name__}({args})"

    def __add__(self, other):
        '''
        Compute addition with dual number or regular number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number being added.
        
        Returns
        ----------
        DualNumber

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real+other, self.dual)
        else:
            return DualNumber(self.real + other.real, self.dual + other.dual)
    
    def __radd__(self, other):
        '''
        Compute reflective addition with regular number.

        Parameters
        ----------
        other : int or float
            Other number being added.
        
        Returns
        ----------
        DualNumber

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        return self.__add__(other)
    
    def __sub__(self, other):
        
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real - other, self.dual)
        else:
            return DualNumber(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other):
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        else:
            return DualNumber(other - self. real, - self.dual)

    def __mul__(self, other):
        '''
        Compute Multiplication with dual number or regular number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number being multiplied.
        
        Returns
        ----------
        DualNumber

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real * other, self.dual * other)
        else:
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
    
    def __rmul__(self, other):
        '''
        Compute reflective multiplication with regular number.

        Parameters
        ----------
        other : int or float
            Other number being added.
        
        Returns
        ----------
        DualNumber

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        return self.__mul__(other)

    def __pow__(self,other):
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        elif isinstance(other, self._supported_scalars):
            return DualNumber(self.real**other, other*self.real**(other-1)*self.dual )
        else:
            return DualNumber(self.real**other.real,self.real**other.real*(other.real*self.dual/self.real+other.dual*np.log(self.real)))
        
    def __neg__(self):
        return DualNumber(- self.real , - self.dual)

    def __truediv__(self, other):
        return self.__mul__(self, other**(-1))

    def __rtruediv__(self, other):
        return self.__mul__(self.__pow__(-1), other)