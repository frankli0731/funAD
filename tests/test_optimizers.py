# -*- coding: utf-8 -*-
"""
This test suite (a module) runs tests for optimizers of the
funAD package.
"""
import pytest
import numpy as np
from funAD import optimizers
from funAD import DualNumber

class TestOptimizers():
    
    def test_init(self):

        lr=0.5
        max_iter=100
        eps=1
        fn = lambda x: lr
        op = optimizers.Optimizer(learning_rate=lr, max_iteration=max_iter, eps=eps)
        assert callable(op.eta) == True and op.eta.__code__.co_code == fn.__code__.co_code
        assert op.max_iteration==max_iter and op.eps==eps

        fn2=lambda x: x*x
        op2 = optimizers.Optimizer(learning_rate=fn2, max_iteration=max_iter, eps=eps)
        assert callable(op2.eta) == True and op2.eta.__code__.co_code == fn2.__code__.co_code
        assert op2.max_iteration==max_iter and op2.eps==eps

    def test_minimize(self):
        lr=0.5
        max_iter=100
        eps=1
        fn = lambda x: lr
        op = optimizers.Optimizer(learning_rate=lr, max_iteration=max_iter, eps=eps)
        def f(x1,x2):
            return (x1-2)**2 + (x2+3)**2 
        with pytest.raises(NotImplementedError):
            op.minimize(f)
        
class TestGD():
    
    def test_minimize(self):
        gd = optimizers.GD()
        def f(x1,x2):
            return (x1-2)**2 + (x2+3)**2 
        x,f_min,history = gd.minimize(f,x_dim=2,verbose=True)
        assert round(x[0],7)==1.9999997 and round(x[1],7)==-2.9999996 and round(f_min,7)==0 #2.4867022e-13

class TestAdam():
    
    #def test_init(self):
    def test_minimize(self):
        adam = optimizers.Adam()
        def f(x1,x2):
            return (x1-2)**2 + (x2+3)**2 
        x,f_min,history = adam.minimize(f,x_dim=2,verbose=True)
        assert round(x[0],6)==2 and round(x[1],7)==-2.9999998 and round(f_min,7)==0 #5.8086916e-14
        
        
