# -*- coding: utf-8 -*-
"""

This module implement overloading functions to handle arithmetic for dual numbers.

"""

import numpy as np
from .dual_number import DualNumber

def sin(self):
  '''
  Overloads the sin() function. 

  Note
  ----------
  Rely on numpy sine function.

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
  if isinstance(self,DualNumber):
    return DualNumber(np.sin(self.real), np.cos(self.real)*self.dual)
  else:
    return np.sin(self)

def cos(self):
  '''
  Overloads the cos() function. 

  Note
  ----------
  Rely on numpy cosine function.

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
  if isinstance(self,DualNumber):
    return DualNumber(np.cos(self.real), -1*np.sin(self.real)*self.dual)
  else:
    return np.cos(self)

def exp(self):
  '''
  Overloads the exponential function. 

  Note
  ----------
  Rely on numpy exp function.

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