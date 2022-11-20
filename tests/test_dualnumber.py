# -*- coding: utf-8 -*-
"""
This test suite (a module) runs tests for dual_number of the
funAD package.
"""
import pytest
import numpy as np
from funAD import DualNumber

class TestDualNumber():
    
    def test_init(self):
        # test init
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        assert d1.real == real1 and d1.dual == dual1
        d2=DualNumber(real1)
        assert d1.real == real1 and d2.dual == 1
        
    def test_add(self):
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

    def test_sub(self):
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
    
    def test_mul(self):
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

    def test_truediv(self):
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
    
    def test_pow(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test power operator (dual number ** dual number)
        d3 = d1**d2
        assert d3.real == 1 and d3.dual == 6 + np.log(real1)

        # test power operator (dual number ** int)
        d3=d1**int_num
        d2/=int_num
        assert d3.real == real1**int_num and d3.dual == dual1**int_num

        # test /= operator (dual number ** float)
        d3=d1**float_num
        assert d3.real == real1**int_num and d3.dual == dual1**int_num

    # reflective operators
    def test_radd(self):
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
        
    def test_rsub(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        int_num=int(1)
        float_num=float(1.0)
        
         # test subtraction (int-dual number)
        d3 = int_num-d1
        assert d3.real == int_num - real1 and d3.dual == -dual1

        # test subtraction (float-dual number)
        d3 = float_num-d1
        assert d3.real == float_num - real1 and d3.dual == -dual1
    
    def test_rmul(self):
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
        
    def test_rtruediv(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test division (int / dual number)
        d3=int_num/d1
        assert d3.real == int_num/real1 and d3.dual == -1*dual1*int_num/(real1*real1)

        # test division (float / dual number)
        d3=float_num/d1
        assert d3.real == float_num/real1 and d3.dual == -1*dual1*float_num/(real1*real1)
    
    def test_iadd(self):
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
    
    def test_rpow(self):
        real1=2
        dual1=1
        d1=DualNumber(real1,dual1)
        
        int_num=int(2)
        float_num=float(2.0)
        
        # test power operator (int ** dual number)
        d3 = int_num ** d1
        assert d3.real == int_num**real1 and d3.dual == np.log(2) * 2 ** 2

        # test power operator (float ** int)
        d3 = float_num ** d1
        assert d3.real == float_num**real1 and d3.dual == np.log(2) * 2 ** 2

    def test_isub(self):
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
        assert d2.real == real2-real1 and d2.dual == dual2-dual1

        # test -= operator (dual number -= int)
        d2=DualNumber(real2,dual2)
        d2-=int_num
        assert d2.real == real2-int_num and d2.dual == dual2

        # test -= operator (dual number -= float)
        d2=DualNumber(real2,dual2)
        d2-=float_num
        assert d2.real == real2-float_num and d2.dual == dual2

    def test_imul(self):
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

    def test_itruediv(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        
        real2=3
        dual2=4
        d2=DualNumber(real2,dual2)
        
        int_num=int(1)
        float_num=float(1.0)
        
        # test /= operator (dual number /= dual number)
        d1/=d2
        assert d1.real == real1/real2 and d1.dual == (dual1*real2-real1*dual2)/(real2*real2)

        # test /= operator (dual number /= int)
        d2=DualNumber(real2,dual2)
        d2/=int_num
        assert d2.real == real2/int_num and d2.dual == dual2

        # test /= operator (dual number /= float)
        d2=DualNumber(real2,dual2)
        d2/=float_num
        assert d2.real == real2/float_num and d2.dual == dual2
    
    # def test_sin(self):
    #     real1=1
    #     dual1=2
    #     d1=DualNumber(real1,dual1)
        
    #     # test sin operator
    #     d2 = DualNumber.sin(d1)
    #     assert d2.real == np.sin(real1) and d2.dual == dual1*np.cos(real1)
        
    # def test_cos(self):
    #     real1=1
    #     dual1=2
    #     d1=DualNumber(real1,dual1)
        
    #     # test cos operator
    #     d2 = DualNumber.cos(d1)
    #     assert d2.real == np.cos(real1) and d2.dual == -1*np.sin(real1)*dual1
        
    # def test_exp(self):
    #     real1=1
    #     dual1=2
    #     d1=DualNumber(real1,dual1)
    #     # test exp operator
    #     d2 = DualNumber.exp(d1)
    #     assert d2.real == np.exp(real1) and d2.dual == np.exp(real1)*dual
    
    def test_neg(self):
        real1=1
        dual1=2
        d1=DualNumber(real1,dual1)
        # test negation operator
        d2 = -d1
        assert d2.real == -real1 and d2.dual == -dual1
    
    # def test_pos(self):
    #     real1=1
    #     dual1=2
    #     d1=DualNumber(real1,dual1)
    #     # test negation operator
    #     d2 = +d1
    #     assert d2.real == real1 and d2.dual == dual1
        
    # def test_eq(self):
    #     real1=1
    #     dual1=2
    #     d1=DualNumber(real1,dual1)
    #     d2=DualNumber(real1,dual1)
        
    #     real3=2
    #     dual3=1
    #     d3=DualNumber(real3,dual3)
        
    #     int_num=int(real1)
    #     float_num=float(real1)
        
    #     # test equality operator for dual numbers
    #     assert d1==d2
    #     assert not (d1==d3)
    #     # test equality operator with int number
    #     assert d1==(int_num)
    #     assert not(d1==int_num-1)
    #     # test equality operator with float number
    #     assert d1==(float_num)
    #     assert not(d1==float_num-1.0)
        
    # def test_neq(self):
    #     real1=1
    #     dual1=2
    #     d1=DualNumber(real1,dual1)
    #     d2=DualNumber(real1,dual1)
        
    #     real3=2
    #     dual3=1
    #     d3=DualNumber(real3,dual3)
        
    #     int_num=int(real1)
    #     float_num=float(real1)
        
    #     # test not equality operator for dual numbers
    #     assert d1!=d3
    #     assert not (d1!=d2)
    #     # test not equality operator with int number
    #     assert d1!=(int_num-1)
    #     assert not(d1!=int_num)
    #     # test not equality operator with float number
    #     assert d1!=(float_num-1.0)
    #     assert not(d1!=float_num)

    # def test_lt(self):
    #     real1=1
    #     dual1=2
    #     d1=DualNumber(real1,dual1)
        
    #     real2=3
    #     dual2=4
    #     d2=DualNumber(real2,dual2)
        
    #     int_num=int(5)
    #     float_num=float(5.0)
        
    #     # test less-than operator for dual numbers
    #     assert d1<d2
    #     # test less-than operator with int number
    #     assert d1<int_num
    #     # test less-than operator with float number
    #     assert d1<float_num
        
    # def test_gt(self):
    #     real1=5
    #     dual1=4
    #     d1=DualNumber(real1,dual1)
        
    #     real2=3
    #     dual2=2
    #     d2=DualNumber(real2,dual2)
        
    #     int_num=int(1)
    #     float_num=float(1.0)
        
    #     # test greater-than operator for dual numbers
    #     assert d1>d2
    #     # test greater-than operator with int number
    #     assert d1>int_num
    #     # test greater-than operator with float number
    #     assert d1>float_num

    # def test_le(self):
    #     real1=1
    #     dual1=2
    #     d1=DualNumber(real1,dual1)
        
    #     real2=3
    #     dual2=4
    #     d2=DualNumber(real2,dual2)
        
    #     real3=1
    #     dual3=2
    #     d3=DualNumber(real3,dual3)
        
    #     int_num=int(5)
    #     int_num2=int(1)
    #     float_num=float(5.0)
    #     float_num2=float(1.0)
        
    #     # test less-than-or-equal operator for dual numbers
    #     assert d1<=d2
    #     assert d1<=d3
    #     # test less-than-or-equal operator with int number
    #     assert d1<=int_num
    #     assert d1<=int_num2
    #     # test less-than-or-equal operator with float number
    #     assert d1<=float_num
    #     assert d1<=float_num2
        
    # def test_le(self):
    #     real1=5
    #     dual1=4
    #     d1=DualNumber(real1,dual1)
        
    #     real2=3
    #     dual2=2
    #     d2=DualNumber(real2,dual2)
        
    #     real3=5
    #     dual3=4
    #     d3=DualNumber(real3,dual3)
        
    #     int_num=int(1)
    #     int_num2=int(5)
    #     float_num=float(1.0)
    #     float_num2=float(5.0)
        
    #     # test greater-than-or-equal operator for dual numbers
    #     assert d1>=d2
    #     assert d1>=d3
    #     # test greater-than-or-equal operator with int number
    #     assert d1>=int_num
    #     assert d1>=int_num2
    #     # test greater-than-or-equal operator with float number
    #     assert d1>=float_num
    #     assert d1>=float_num2
        
if __name__ == "__main__":
    pass
