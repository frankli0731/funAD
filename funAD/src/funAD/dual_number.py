

class DualNumber(object):
    _supported_scalars = (int, float)
    
    def __init__(self,real,dual = None):
        self.real = real
        if dual == None:
            self.dual = 1
        else:
            self.dual = dual

    def __add__(self, other):
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real+other, self.dual)
        else:
            return DualNumber(self.real + other.real,self.dual+other.dual)
    
    def __mul__(self, other):
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsuported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real * other, self.dual * other)
        else:
            return DualNumber(self.real * other.real, self.real*other.dual+self.dual*other.real)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def set_dual(self,dual):
      self.dual = dual