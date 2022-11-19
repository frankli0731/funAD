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

        Parameters
        ----------
        real : float
            Real part of dual number.
        dual : float
            Dual part of dual number, default to 1 if not specified.
        
        Note
        ----
        Dual part is typically only set to 1 through external function
        when finding derivative with respect to that dual number.

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
        
        Raises
        ------
        TypeError
        If the other number inputted is not of any supported numeric format.     

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
            Other number being added to.
        
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
        '''
        Compute Subtraction of dual number or regular number from dual number or regular number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number being subtracted.
        
        Returns
        ----------
        DualNumber
        
        Raises
        ------
        TypeError
        If the other is not of any supported numeric format.        

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
            return DualNumber(self.real - other, self.dual)
        else:
            return DualNumber(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other):
        '''
        Compute Subtraction of dual number from regular number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number being subtracted from.
        
        Returns
        ----------
        DualNumber
        
        Raises
        ------
        TypeError
        If the other is not of any supported numeric format.        

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        if not isinstance(other, self._supported_scalars):
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
        -------
        DualNumber
        
        Raises
        ------
        TypeError
        If the other number inputted is not of any supported numeric format.      

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
            Other number being multiplied.
        
        Returns
        -------
        DualNumber
        
        Raises
        ------
        TypeError
        If the other number input is not of any supported numeric format.     

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        return self.__mul__(other)

    def __pow__(self,other):
        '''
        Compute power raised to input regular number or dual number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number as the power a dual number is being raised to.
        
        Returns
        -------
        DualNumber
        
        Raises
        ------
        TypeError
        If the other number input is not of any supported numeric format.     

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        elif isinstance(other, self._supported_scalars):
            return DualNumber(self.real**other, other*self.real**(other-1)*self.dual)
        else:
            return DualNumber(self.real**other.real,self.real**other.real*(other.real*self.dual/self.real+other.dual*np.log(self.real)))
        
    def __neg__(self):
        '''
        Change the sign of input dual number by multiplying real and dual part with -1.
        
        Returns
        -------
        DualNumber 

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        return DualNumber(- self.real , - self.dual)

    def __truediv__(self, other):
        '''
        Compute float division of dual number by int, float or dual number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number being divided.
        
        Returns
        -------
        DualNumber
        
        Raises
        ------
        ZeroDivisionError
            If the denominator other number's real part is zero.

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        elif isinstance(other, self._supported_scalars):
            return DualNumber(self.real/other, self.dual/other)
        else:
            return DualNumber(self.real/other.real, (self.dual*other.real - other.dual*self.real)/(other.real*other.real))

    def __rtruediv__(self, other):        
        '''
        Compute float division of real number by dual number.

        Parameters
        ----------
        other : int or float
            Other real number being divided.
        
        Returns
        -------
        DualNumber
        
        Raises
        ------
        ZeroDivisionError
            If the denominator dual number's real part is zero.

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case

        '''
        if isinstance(other, self._supported_scalars):
            return DualNumber(other/self.real, (-1*other*self.dual)/(self.real*self.real))    
        else:
            raise TypeError(f"Unsuported type '{type(other)}'")

    #=================== inplace operation ===================#
    def __iadd__(self,other):
        '''
        Compute inplace addition of dual number by int, float or dual number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number being added.
        
        Returns
        -------
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
        elif isinstance(other, self._supported_scalars):
            self.real += other
            return self
        else:
            self.real += other.real
            self.dual += other.dual
            return self

    def __isub__(self,other):
        '''
        Compute inplace substraction of dual number by int, float or dual number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number being added.
        
        Returns
        -------
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
        elif isinstance(other, self._supported_scalars):
            self.real -= other
            return self
        else:
            self.real -= other.real
            self.dual -= other.dual
            return self
    
    def __imul__(self,other):
        '''
        Compute inplace multiplication of dual number by int, float or dual number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number being added.
        
        Returns
        -------
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
        elif isinstance(other, self._supported_scalars):
            self.real *= other
            self.dual *= other
            return self
        else:
            self.real,self.dual = self.real*other.real, self.dual*other.real + self.real*other.dual
            return self
    
    def __itruediv__(self,other):
        '''
        Compute inplace division of dual number by int, float or dual number.

        Parameters
        ----------
        other : int or float or DualNumber instance
            Other number being added.
        
        Returns
        -------
        DualNumber
        
        Raises
        ------
        ZeroDivisionError
            If the denominator other number's real part is zero.

        Examples
        --------
        Please insert test case

        >>> Please insert test case
        >>> Please insert test case
        Please insert test case
        '''
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        elif isinstance(other, self._supported_scalars):
            self.real /= other
            self.dual /= other
            return self
        else:
            self.real,self.dual = self.real/other.real, (self.dual*other.real - self.real*other.dual)/(other.real*other.real)
            return self


######################################

# -*- coding: utf-8 -*-
"""

This module implement overloading functions to handle arithmetic for dual numbers.

"""

import numpy as np

def exp(self):
  '''
  Overloads the exponential function. 

  Parameters
  ----------
  x : int or float or DualNumber instance

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Please insert test case

  >>> Please insert test case
  >>> Please insert test case
  Please insert test case   

  '''
  if isinstance(self,DualNumber):
    return DualNumber(np.exp(self.real), np.exp(self.real)*self.dual)
  else:
    return np.exp(self)

def log(self):
  '''
  Overloads the natural logarithm function. 

  Parameters
  ----------
  x : int or float or DualNumber instance

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Please insert test case

  >>> Please insert test case
  >>> Please insert test case
  Please insert test case   

  '''
  if isinstance(self,DualNumber):
    return DualNumber(np.log(self.real), self.dual/self.real)
  else:
    return np.log(self)

#=================== Trigonometric ===================#
def sin(x):
  '''
  Overloads the sin() function. 

  Parameters
  ----------
  x : int or float or DualNumber instance
      Each number represent angle in radians (:math:`2 \pi` rad equals 360 degrees).

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Print sine of one angle:

  >>> ad.sin(np.pi/2.)
  1.0

  >>> x = DualNumber(np.pi/2)
  >>> ad.sin(x)
  DualNumber(real = 1., dual = 0.)
  '''
  if isinstance(x,DualNumber):
    return DualNumber(np.sin(x.real), np.cos(x.real)*x.dual)
  else:
    return np.sin(x)

def cos(x):
  '''
  Overloads the cos() function. 

  Parameters
  ----------
  x : int or float or DualNumber instance
      Each number represent angle in radians (:math:`2 \pi` rad equals 360 degrees).

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Print cosine of one angle:

  >>> ad.cos(np.pi/2.)
  0.0

  >>> x = DualNumber(np.pi/2)
  >>> ad.cos(x)
  DualNumber(real = 0., dual = -1.)
  '''
  if isinstance(x,DualNumber):
    return DualNumber(np.cos(x.real), -1*np.sin(x.real)*x.dual)
  else:
    return np.cos(x)

def tan(x):  
  '''
  Overloads the tangent function. 

  Parameters
  ----------
  x : int or float or DualNumber instance

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Please insert test case

  >>> Please insert test case
  >>> Please insert test case
  Please insert test case   
  '''
  if isinstance(x,DualNumber):
    return DualNumber(np.tan(x.real), x.dual/(np.cos(x.real)**2))
  else:
    return np.tan(x)
  
def arcsin(x):
  '''
  Overloads the inverse sine function. 

  Parameters
  ----------
  x : int or float or DualNumber instance

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Please insert test case

  >>> Please insert test case
  >>> Please insert test case
  Please insert test case   
  '''
  if isinstance(x,DualNumber):
    return DualNumber(np.arcsin(x.real),x.dual/np.sqrt(1-x.real**2))
  else:
    return np.arcsin(x)

def arccos(x):
  '''
  Overloads the inverse cosine function. 

  Parameters
  ----------
  x : int or float or DualNumber instance

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Please insert test case

  >>> Please insert test case
  >>> Please insert test case
  Please insert test case   
  '''
  if isinstance(x,DualNumber):
    return DualNumber(np.arccos(x.real),-1*x.dual/np.sqrt(1-x.real**2))
  else:
    return np.arccos(x)

def arctan(x):
  '''
  Overloads the inverse tangent function. 

  Parameters
  ----------
  x : int or float or DualNumber instance

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Please insert test case

  >>> Please insert test case
  >>> Please insert test case
  Please insert test case   
  '''
  if isinstance(x,DualNumber):
    return DualNumber(np.arctan(x.real),x.dual/(1+x.real**2))
  else:
    return np.arctan(x)

#=================== Hyperbolic ===================#
def sinh(x):
  '''
  Overloads the hyperbolic sine function. 

  Parameters
  ----------
  x : int or float or DualNumber instance

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Please insert test case

  >>> Please insert test case
  >>> Please insert test case
  Please insert test case   
  '''
  if isinstance(x,DualNumber):
    return DualNumber(np.sinh(x.real),x.dual*np.cosh(x.real))
  else:
    return np.sinh(x)  

def cosh(x):
  '''
  Overloads the hyperbolic cosine function. 

  Parameters
  ----------
  x : int or float or DualNumber instance

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Please insert test case

  >>> Please insert test case
  >>> Please insert test case
  Please insert test case   
  '''
  if isinstance(x,DualNumber):
    return DualNumber(np.cosh(x.real),x.dual*np.sinh(x.real))
  else:
    return np.cosh(x) 

def tanh(x):
  '''
  Overloads the hyperbolic tangent function. 

  Parameters
  ----------
  x : int or float or DualNumber instance

  Returns
  -------
  float or DualNumber instance
      
  Examples
  --------
  Please insert test case

  >>> Please insert test case
  >>> Please insert test case
  Please insert test case   
  '''
  if isinstance(x,DualNumber):
    return DualNumber(np.tanh(x.real),x.dual/np.cosh(x.real)**2)
  else:
    return np.tanh(x) 






#########################################################################################################
# -*- coding: utf-8 -*-
"""
This module implements function class, a key component for forward mode autodifferentiation.
"""
from collections.abc import Iterable 
import numpy as np

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

    def __init__(self, *f, x_dim=1):
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
        if len(f) == 1:
            self.f = f[0]
        else:
            self.f = lambda *x: [f(*x) for f in f]
        self.x_dim = x_dim

    def __call__(self,*x):
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
        return self.f(*x)

    def grad(self,*x):
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
        self.array_type_input = False

        if len(x) == 1: #single input
            if isinstance(x[0],Iterable): # if the input is iterable, has len() function
                self.array_type_input = True
                x = x[0]
                if len(x) != self.x_dim:
                    raise ValueError('Dimension Mismatch')
            else: # sinle input x
                if self.x_dim != 1:
                    raise ValueError('Dimension Mismatch')

        else: #multiple input
            if len(x)!=self.x_dim :
                raise ValueError('Dimension Mismatch')

        J = []
        for i in range(self.x_dim):  
            p = np.identity(self.x_dim)[:,i].tolist()
            J.append(self._grad(x,p))
        if len(J) == 1:
            return J[0]
        else:
            return np.array(J).T

    def _grad(self,x,p):
        dual_nums = [DualNumber(*input) for input in zip(x,p)]
        if self.array_type_input == True:
            result = self.f(dual_nums)
        else:
            result = self.f(*dual_nums)
        if isinstance(result,DualNumber):
            return result.dual
        else:
            return np.array([d.dual for d in result])





if __name__=="__main__":
    #R1  to R1
    def f(x1):
        return sin(exp(x1)) + 6 * (cos(x1)) / x1 - 2 * x1
    f = function(f)
    print("1=========")
    print(f(1))
    print(f.grad(1))

    def f(x):
        return sin(exp(x[0])) + 6 * (cos(x[0])) / x[0] - 2 * x[0]
    f = function(f)
    print("2=========")
    print(f([1]))
    print(f.grad([1]))

    # R2 to R1
    def f(x1,x2):
        return sin(exp(x1)) + 6 * (cos(x2)) / x1 - 2 * x2
    f = function(f,x_dim=2)
    print("3=========")
    print(f(1,1))
    print(f.grad(1,1))

    def f(x):
        return sin(exp(x[0])) + 6 * (cos(x[1])) / x[0] - 2 * x[1]
    f = function(f,x_dim=2)
    print("4=========")
    print(f([1,1]))
    print(f.grad([1,1]))


    # R1 to R2
    def f(x1):
        return sin(exp(x1)), 6 * (cos(x1)) / x1 - 2 * x1
    f = function(f,x_dim=1)
    print("5=========")
    print(f(1))
    print(f.grad(1))


    # R1 to R2
    def f(x1):
        return sin(exp(x1))
    def g(x1):
        return 6 * (cos(x1)) / x1 - 2 * x1
    f = function(f,g,x_dim=1)
    print("6=========")
    print(f(1))
    print(f.grad(1))


    # R2 to R2
    def f(x1,x2):
        return sin(exp(x1))
    def g(x1,x2):
        return 6 * (cos(x2)) / x1 - 2 * x2
    f = function(f,g,x_dim=2)
    print("7=========")
    print(f(1,1))
    print(f.grad(1,1))

    def f(x):
        return sin(exp(x[0]))
    def g(x):
        return 6 * (cos(x[1])) / x[0] - 2 * x[1]
    f = function(f,g,x_dim=2)
    print("8=========")
    print(f([1,1]))
    print(f.grad([1,1]))

