import numpy as np
from .dual_number import DualNumber

def sin(self):
  if isinstance(self,DualNumber):
    return DualNumber(np.sin(self.real), np.cos(self.real)*self.dual)
  else:
    return np.sin(self)

def cos(self):
  if isinstance(self,DualNumber):
    return DualNumber(np.cos(self.real), -1*np.sin(self.real)*self.dual)
  else:
    return np.cos(self)

def exp(self):
  if isinstance(self,DualNumber):
    return DualNumber(np.exp(self.real), np.exp(self.real)*self.dual)
  else:
    return np.exp(self)