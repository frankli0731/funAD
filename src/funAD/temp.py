# please ignore, just a temporary test file
import numpy as np
import operations as ad
from function import function
from dual_number import DualNumber

# def f(x):
#     f1 = x[0]+ad.sin(x[1])
#     f2 = x[0]*ad.exp(x[1])
#     return [f1,f2] obsolate implementation

if __name__ == "__main__":
    f = function(lambda x :x[0]+ad.sin(x[1]), lambda x: x[0]*ad.exp(x[1]),x_dim=2)
    x = np.ones((2,))
    print(f.f)

    fx = f(x)
    g = f.grad(x)

    print("value:",fx)
    print("gradient/Jacobian",g)

    # a = DualNumber(1,2)
    # b = DualNumber(2,3)
    # c = 1
    # c /= a
    # a *= b
    # print(a)
    # print(c)
