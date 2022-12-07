from abc import abstractmethod
import numpy as np
from .function import function


class Optimizer():
    def __init__(self, learning_rate = 0.001, max_iteration = 10000, eps = 1e-15):
        if callable(learning_rate):
            self.eta = learning_rate
        else:
            self.eta = lambda t : learning_rate
        self.max_iteration = max_iteration
        self.eps = eps

    @abstractmethod
    def minimize(self,f, x_dim = 1, x0 = None,verbose=False):
        raise NotImplementedError

class GD(Optimizer):
    def minimize(self,f, x_dim = 1, x0 = None,verbose=False):
        if isinstance(f,function):
            if len(f.function_list) > 1: # only functions with a single output can be minimized
                raise TypeError("Cannot optimize vector-valued function")
        else:
            f = function(f,x_dim = x_dim)
        if x0 is None:
            x0 = np.zeros(f.x_dim) # initialize the starting point to be zero (vector)
        x = x0
        t = 0
        history = []
        for i in range(self.max_iteration):
            x_new = x - self.eta(t)*f.grad(x) # update rule of gradient descent
            if abs(f(x)-f(x_new)) < self.eps: # break the loop when the change in function is less than eps
                x = x_new
                break
            x = x_new
            t += 1
            if verbose:
                history.append((x,f(x)))
        if verbose:
            return x,f(x),history
        else:
            return x,f(x)

    def maximize(self,f, x_dim = 1, x0 = None,verbose=False):
        if isinstance(f,function):
            if len(f.function_list) > 1: # only functions with a single output can be maximized
                raise TypeError("Cannot optimize vector-valued function")
            neg_f = function(lambda *x: -1*f.function_list[0](*x),x_dim=x_dim) # build function object neg_f whose "f" attribute equal to -f
        else:
            neg_f = function(lambda *x: -1*f(*x),x_dim = x_dim) # build function object neg_f whose "f" attribute equal to -f
        if verbose:
            x,neg_f_min,history = self.minimize(neg_f,x_dim=x_dim,x0=x0,verbose=True) # recall max f = - min(-f); minimizing neg_f gives -max(f) and argmax(f) 
            history = [(tup[0],-1*tup[1]) for tup in history] # correct history which evaluated -f(x)
            return x,-1*neg_f_min,history
        x,neg_f_min = self.minimize(neg_f,x_dim=x_dim,x0=x0,verbose=False)
        return x,-1*neg_f_min

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, max_iteration = 10000, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08):
        super().__init__(learning_rate, max_iteration)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def minimize(self,f, x_dim = 1, x0 = None,verbose=False):
        if isinstance(f,function):
            if len(f.function_list) > 1: # only functions with a single output can be minimized
                raise TypeError("Cannot optimize vector-valued function")
        else:
            f = function(f,x_dim = x_dim)
        if x0 is None:
            x0 = np.zeros(f.x_dim) # initialize the starting point to be zero (vector)
        x = x0
        t = 0
        m = 0
        v = 0
        history = []
        for i in range(self.max_iteration):
            g = f.grad(x)
            m = self.beta_1 * m + (1-self.beta_1)*g
            v = self.beta_2 * v + (1-self.beta_2)*g**2
            m_hat = m/(1-self.beta_1**(t+1))
            v_hat = v/(1-self.beta_2**(t+1))
            x_new = x - self.eta(t)/np.sqrt(v_hat+self.epsilon)*m_hat # update rule of adam
            if abs(f(x)-f(x_new)) < self.eps: # break the loop when the change in function is less than eps
                x = x_new
                break
            x = x_new
            t += 1
            if verbose:
                    history.append((x,f(x)))
        if verbose:
            return x,f(x), history
        else:
            return x,f(x)
    
    def maximize(self,f, x_dim = 1, x0 = None,verbose=False):
        if isinstance(f,function):
            if len(f.function_list) > 1: # only functions with a single output can be maximized
                raise TypeError("Cannot optimize vector-valued function")
            neg_f = function(lambda *x: -1*f.function_list[0](*x),x_dim=x_dim) # build function object neg_f whose "f" attribute equal to -f
        else:
            neg_f = function(lambda *x: -1*f(*x),x_dim = x_dim) # build function object neg_f whose "f" attribute equal to -f
        if verbose:
            x,neg_f_min,history = self.minimize(neg_f,x_dim=x_dim,x0=x0,verbose=True) # recall max f = - min(-f); minimizing neg_f gives -max(f) and argmax(f) 
            history = [(tup[0],-1*tup[1]) for tup in history] # correct history which evaluated -f(x)
            return x,-1*neg_f_min,history
        x,neg_f_min = self.minimize(neg_f,x_dim=x_dim,x0=x0,verbose=False)
        return x,-1*neg_f_min


