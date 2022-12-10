# -*- coding: utf-8 -*-
"""
This test suite (a module) runs tests for optimizers of the
funAD package.
"""
import pytest
import numpy as np
from funAD import optimizers
from funAD import DualNumber
from funAD import function

class TestOptimizers():
    
    def test_init(self):
        lr=0.5
        max_iter=100
        eps=1
        fn = lambda x: lr
        op = optimizers.Optimizer(learning_rate=lr, max_iteration=max_iter, eps=eps)
        assert callable(op.eta) == True and op.eta.__code__.co_code == fn.__code__.co_code
        assert op.max_iteration==max_iter and op.eps==eps and op.lazy==False

        fn2=lambda x: x*x
        op2 = optimizers.Optimizer(learning_rate=fn2, max_iteration=max_iter, eps=eps)
        assert callable(op2.eta) == True and op2.eta.__code__.co_code == fn2.__code__.co_code
        assert op2.max_iteration==max_iter and op2.eps==eps and op.lazy==False

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

        # when verbose is set to be True
        x,f_min,history = gd.minimize(f,x_dim=2,verbose=True)
        assert round(x[0],1)==2.0 and round(x[1],1)==-3.0 and round(f_min,7)==0 and type(history)==list

        # when verbose is set to be False
        assert len(gd.minimize(f,x_dim=2,verbose=False))==2
        x,f_min = gd.minimize(f,x_dim=2,verbose=False)
        assert round(x[0],1)==2.0 and round(x[1],1)==-3.0 and round(f_min,7)==0 #2.4867022e-13

        # multiple outputs case
        def f2(x1):
            return x1**2, x1+2
        multi_fcn = function(f,f2)
        with pytest.raises(TypeError):
            gd.minimize(multi_fcn ,x_dim=2,verbose=True)

        # when lazy is set to be True
        gd = optimizers.GD(lazy=True)
        x,f_min,history = gd.minimize(f,x_dim=2,verbose=True)
        assert round(x[0],1)==2.0 and round(x[1],1)==-3.0 and round(f_min,7)==0 and type(history)==list

    def test_maximize(self):
        gd = optimizers.GD()
        def f(x1,x2):
            return (x1-2)**2 + (x2+3)**2 
        def f2(x1):
            return x1**2, x1+2

        # when verbose is set to be True
        assert len(gd.maximize(f,x_dim=2,verbose=True))==3
        x,f_max,history = gd.maximize(f,x_dim=2,verbose=True)
        assert round(x[0],4)==-951141885.2122  and round(x[1],4)==1426712827.8182 and f_max == 2.94018039123093e+18  and type(history)==list
        
        # when verbose is set to be False
        assert len(gd.maximize(f,x_dim=2,verbose=False))==2
        x,f_max = gd.maximize(f,x_dim=2,verbose=False)
        assert round(x[0],4)==-951141885.2122  and round(x[1],4)==1426712827.8182 and f_max == 2.94018039123093e+18 
        
        # multiple outputs case
        multi_fcn = function(f,f2)
        with pytest.raises(TypeError):
            gd.maximize(multi_fcn ,x_dim=2,verbose=True)
        
        
class TestAdam():
    
    def test_minimize(self):
        adam = optimizers.Adam()
        def f(x1,x2):
            return (x1-2)**2 + (x2+3)**2
        
        # when verbose is set to be True
        x,f_min,history = adam.minimize(f,x_dim=2,verbose=True)
        assert round(x[0],1)==2.0 and round(x[1],1)==-3.0 and round(f_min,7)==0 and type(history)==list

        # when verbose is set to be False
        assert len(adam.minimize(f,x_dim=2,verbose=False))==2
        x,f_min = adam.minimize(f,x_dim=2,verbose=False)
        assert round(x[0],1)==2.0 and round(x[1],1)==-3.0 and round(f_min,7)==0
        
        # multiple outputs case
        def f2(x1):
            return x1**2, x1+2
        multi_fcn = function(f,f2)
        with pytest.raises(TypeError):
            adam.minimize(multi_fcn ,x_dim=2,verbose=True)

        # when lazy is set to be True
        adam = optimizers.Adam(lazy=True)
        x,f_min,history = adam.minimize(f,x_dim=2,verbose=True)
        assert round(x[0],1)==2.0 and round(x[1],1)==-3.0 and round(f_min,7)==0 and type(history)==list


    def test_maximize(self):
        adam = optimizers.Adam()
        def f(x1,x2):
            return (x1-2)**2 + (x2+3)**2 
        def f2(x1):
            return x1**2, x1+2

        # when verbose is set to be True
        assert len(adam.maximize(f,x_dim=2,verbose=True))==3
        x,f_max,history = adam.maximize(f,x_dim=2,verbose=True)
        assert round(x[0],4)==-11.4134 and round(x[1],4)==11.1903 and round(f_max,4) == 381.2838  and type(history)==list
        
        # when verbose is set to be False
        assert len(adam.maximize(f,x_dim=2,verbose=False))==2
        x,f_max = adam.maximize(f,x_dim=2,verbose=False)
        assert round(x[0],4)==-11.4134 and round(x[1],4)==11.1903 and round(f_max,4) == 381.2838 
        
        # multiple outputs case
        multi_fcn = function(f,f2)
        with pytest.raises(TypeError):
            adam.maximize(multi_fcn ,x_dim=2,verbose=True)
