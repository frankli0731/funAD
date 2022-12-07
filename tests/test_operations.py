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
        d3 = op.cos(real1)
        assert d3 == np.cos(real1)
        
    def test_exp(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test exp operator
        d2 = op.exp(d1)
        assert d2.real == np.exp(real1) and d2.dual == np.exp(real1)*dual1
        
    def test_log(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test log operator
        d2 = op.log(d1)
        assert d2.real == np.log(real1) and d2.dual == 1/real1*dual1
        d3 = op.log(real1)
        assert d3 == np.log(real1)

    def test_tan(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test tan operator
        d2 = op.tan(d1)
        assert d2.real == np.tan(real1) and d2.dual == 1/np.cos(real1)**2*dual1
        d3 = op.tan(real1)
        assert d3 == np.tan(real1)

    def test_arcsin(self):
        real1=1/2
        dual1=2
        d1=DualNumber(real1,dual1)
        # test arcsin operator
        d2 = op.arcsin(d1)
        assert d2.real == np.arcsin(real1) and d2.dual == 1/np.sqrt(1-real1**2)*dual1
        d3 = op.arcsin(real1)
        assert d3 == np.arcsin(real1)

    def test_arccos(self):
        real1=1/2
        dual1=2
        d1=DualNumber(real1,dual1)
        # test arccos operator
        d2 = op.arccos(d1)
        assert d2.real == np.arccos(real1) and d2.dual == -1/np.sqrt(1-real1**2)*dual1
        d3 = op.arccos(real1)
        assert d3 == np.arccos(real1)

    def test_arctan(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test arctan operator
        d2 = op.arctan(d1)
        assert d2.real == np.arctan(real1) and d2.dual == 1/(1+real1**2)*dual1
        d3 = op.arctan(real1)
        assert d3 == np.arctan(real1)

    def test_sinh(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test sinh operator
        d2 = op.sinh(d1)
        assert d2.real == np.sinh(real1) and d2.dual == np.cosh(real1)*dual1
        d3 = op.sinh(real1)
        assert d3 == np.sinh(real1)

    def test_cosh(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test cosh operator
        d2 = op.cosh(d1)
        assert d2.real == np.cosh(real1) and d2.dual == np.sinh(real1)*dual1
        d3 = op.cosh(real1)
        assert d3 == np.cosh(real1)

    def test_tanh(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test tanh operator
        d2 = op.tanh(d1)
        assert d2.real == np.tanh(real1) and d2.dual == 1/np.cosh(real1)**2*dual1
        d3 = op.tanh(real1)
        assert d3 == np.tanh(real1)

    def test_log(self):
        real1=7
        dual1=8
        d1=DualNumber(real1,dual1)
        
        # test log operator
        # natural log
        d2 = op.log(d1)
        assert d2.real == np.log(real1) and d2.dual == dual1/real1
        # base specified
        d3 = op.log(d2,base=1.1)
        assert d2.real == np.log(real1) and d2.dual == dual1/real1

if __name__ == "__main__":
    pass
