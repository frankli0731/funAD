import numpy as np
from dual_number import DualNumber

class function:

    def __init__(self, f=None):
        self.f = f

    def __call__(self,x):
        return self.f(x)

    def grad(self,x,p=None):
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