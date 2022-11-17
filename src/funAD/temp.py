# please ignore, just a temporary test file
import numpy as np
import operations as ad
from function import function
from dual_number import DualNumber

def f(x):
    f1 = x[0]+ad.sin(x[1])
    f2 = x[0]*ad.exp(x[1])
    return f1

if __name__ == "__main__":
    f = function(f)
    x = np.ones((2,))

    fx = f(x)
    g = f.grad(x)

    print("value:",fx)
    print("gradient/Jacobian",g)
