# -*- coding: utf-8 -*-
"""
This test suite (a module) runs tests for dual_number of the
funAD package.
"""

import pytest
import unittest

from funAD.dual_number import DualNumber

def test_class_DualNumber(self):
    
    def test_init():
        # test init
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        assert d1.real == real1 and d1.dual == dual1
        
    def test_add():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        # test addition (dual number + dual number)
        d3=d1+d2
        assert d3.real == real1+real2 and d3.dual == dual1+dual2

        # test addition (dual number + int)
        int_num=int(1)
        d3 = d1+int_num
        assert d3.real == real1+int_num and d3.dual == dual1

        # test addition (dual number + float)
        float_num=float(1.0)
        d3 = d1+float_num
        assert d3.real == real1+float_num and d3.dual == dual1

    def test_sub():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        # test subtraction (dual number - dual number)
        d3=d1-d2
        assert d3.real == real1-real2 and d3.dual == dual1-dual2

        # test subtraction (dual number - int)
        int_num=int(1)
        d3 = d1-int_num
        assert d3.real == real1-int_num and d3.dual == dual1

        # test subtraction (dual number - float)
        float_num=float(1.0)
        d3 = d1-float_num
        assert d3.real == real1-float_num and d3.dual == dual1
    
    def test_mul():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test multiplication (dual number * dual number)
        d3=d1*d2
        assert d3.real == real1*real2 and d3.dual == real1*dual2+real2*dual1

        # test multiplication (dual number * int)
        d3=d1*int_num
        assert d3.real == real1*int_num and d3.dual == dual1

        # test multiplication (dual number * float)
        d3=d1*float_num
        assert d3.real == real1*float_num and d3.dual == dual1

    def test_truediv():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test division (dual number / dual number)
        d3=d1/d2
        assert d3.real == real1/real2 and d3.dual == (dual1*real2-real1*dual2)/(real2*real2)

        # test division (dual number / int)
        d3=d1/int_num
        assert d3.real == real1/int_num and d3.dual == dual1

        # test division (dual number / float)
        d3=d1/float_num
        assert d3.real == real1/float_num and d3.dual == dual1

    # reflective operators
    def test_radd():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test addition (int+dual number)
        d3 = int_num+d1
        assert d3.real == real1+int_num and d3.dual == dual1

        # test addition (float+dual number)
        float_num=float(1.0)
        d3 = float_num+d1
        assert d3.real == real1+float_num and d3.dual == dual1
        
    def test_rsub():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        int_num=int(1)
        float_num=float(1.0)
        
         # test subtraction (int-dual number)
        d3 = int_num-d1
        assert d3.real == real1-int_num and d3.dual == dual1

        # test subtraction (float-dual number)
        d3 = float_num-d1
        assert d3.real == real1-float_num and d3.dual == dual1
    
    def test_rmul():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test multiplication (int * dual number)
        d3=int_num*d1
        assert d3.real == real1*int_num and d3.dual == dual1

        # test multiplication (float * dual number)
        d3=float_num*d1
        assert d3.real == real1*float_num and d3.dual == dual1
        
    def test_rtruediv():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test division (int / dual number)
        d3=int_num/d1
        assert d3.real == int_num/real1 and d3.dual == dual1

        # test division (float / dual number)
        d3=float_num/d1
        assert d3.real == float_num/real1 and d3.dual == dual1
    
    def test_iadd():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test += operator (dual number += dual number)
        d2+=d1
        assert d2.real == real1+real2 and d2.dual == dual1+dual2

        # test += operator (dual number += int)
        d2=DualNumber(real2,dual2)
        d2+=int_num
        assert d2.real == real2+int_num and d2.dual == dual2

        # test += operator (dual number += float)
        d2=DualNumber(real2,dual2)
        d2+=float_num
        assert d2.real == real2+float_num and d2.dual == dual2

    def test_isub():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test -= operator (dual number -= dual number)
        d2-=d1
        assert d2.real == real1-real2 and d2.dual == dual1-dual2

        # test -= operator (dual number -= int)
        d2=DualNumber(real2,dual2)
        d2-=int_num
        assert d2.real == real2-int_num and d2.dual == dual2

        # test -= operator (dual number -= float)
        d2=DualNumber(real2,dual2)
        d2-=float_num
        assert d2.real == real2-float_num and d2.dual == dual2

    def test_imul():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test *= operator (dual number *= dual number)
        d2*=d1
        assert d2.real == real1*real2 and d2.dual == real1*dual2+real2*dual1

        # test *= operator (dual number *= int)
        d2=DualNumber(real2,dual2)
        d2*=int_num
        assert d2.real == real2*int_num and d2.dual == dual2

        # test *= operator (dual number *= float)
        d2=DualNumber(real2,dual2)
        d2*=float_num
        assert d2.real == real2*float_num and d2.dual == dual2

    def test_itruediv():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        int_num=int(1)
        float_num=float(1.0)
        
         # test /= operator (dual number /= dual number)
        d2/=d1
        assert d2.real == real1/real2 and d2.dual == (dual1*real2-real1*dual2)/(real2*real2)

        # test /= operator (dual number /= int)
        d2=DualNumber(real2,dual2)
        d2/=int_num
        assert d2.real == real2/int_num and d2.dual == dual2

        # test /= operator (dual number /= float)
        d2=DualNumber(real2,dual2)
        d2/=float_num
        assert d2.real == real2/float_num and d2.dual == dual2
    
    def test_sin():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        # test sin operator
        d2 = DualNumber.sin(d1)
        assert d2.real == np.sin(real1) and d2.dual == dual1*np.cos(real1)
        
    def test_cos():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        # test cos operator
        d2 = DualNumber.cos(d1)
        assert d2.real == np.cos(real1) and d2.dual == -1*np.sin(real1)*dual1
        
    def test_exp():
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test exp operator
        d2 = DualNumber.exp(d1)
        assert d2.real == np.exp(real1) and d2.dual == np.exp(real1)*dual
