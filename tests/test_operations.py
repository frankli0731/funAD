# -*- coding: utf-8 -*-
"""
This test suite (a module) runs tests for dual_number of the
funAD package.
"""
import pytest
import numpy as np
from funAD import operations as op
from funAD import DualNumber

class TestOperations():
   
    def test_sin(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        # test sin operator
        d2 = op.sin(d1)
        assert d2.real == np.sin(real1) and d2.dual == dual1*np.cos(real1)
        
    def test_cos(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        # test cos operator
        d2 = op.cos(d1)
        assert d2.real == np.cos(real1) and d2.dual == -1*np.sin(real1)*dual1
        
    def test_exp(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test exp operator
        d2 = op.exp(d1)
        assert d2.real == np.exp(real1) and d2.dual == np.exp(real1)*dual1
    

if __name__ == "__main__":
    pass
