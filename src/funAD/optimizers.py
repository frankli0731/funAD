from abc import abstractmethod
import numpy as np
from .function import function


class Optimizer():
    def __init__(self, learning_rate = 0.001, max_iteration = 10000):
        if callable(learning_rate):
            self.eta = learning_rate
        else:
            self.eta = lambda t : learning_rate
        self.max_iteration = max_iteration

    @abstractmethod
    def minimize(self,f, x_dim = 1, x0 = None,verbose=False):
        raise NotImplementedError

class GD(Optimizer):
    def minimize(self,f, x_dim = 1, x0 = None,verbose=False):
        if isinstance(f,function):
            if len(f.function_list) > 1:
                raise TypeError("Cannot optimize vector-valued function")
        else:
            f = function(f,x_dim = x_dim)
        if x0 is None:
            x0 = np.zeros(f.x_dim)
        x = x0
        t = 0
        history = []
        for i in range(self.max_iteration):
            x = x - self.eta(t)*f.grad(x0)
            t += 1
            if verbose:
                history.append((x,f(x)))
        if verbose:
            return x,f(x),history
        else:
            return x,f(x)

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, max_iteration = 10000, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07):
        super().__init__(learning_rate, max_iteration)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def minimize(self,f, x_dim = 1, x0 = None,verbose=False):
        if isinstance(f,function):
            if len(f.function_list) > 1:
                raise TypeError("Cannot optimize vector-valued function")
        else:
            f = function(f,x_dim = x_dim)
        if x0 is None:
            x0 = np.zeros(f.x_dim)
        x = x0
        t = 0
        m = 0
        v = 0
        history = []
        for i in range(self.max_iteration):
            g = f.grad(x)
            m = self.beta_1 * m + (1-self.beta_1)*g
            v = self.beta_2 * v + (1-self.beta_2)*g**2
            m_hat = m/(1-self.beta_1)
            v_hat = v/(1-self.beta_2)
            x = x - self.eta(t)/np.sqrt(v_hat+self.epsilon)*m_hat
            t += 1
            if verbose:
                    history.append((x,f(x)))
        if verbose:
            return x,f(x), history
        else:
            return x,f(x)

