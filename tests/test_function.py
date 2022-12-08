import funAD
import pytest
import funAD.operations as op

class TestFunction():
    
    def test_init(self):
       
        # R1
        def fcn_R1(x1):
            return funAD.sin(funAD.exp(x1)) + 6 * (funAD.cos(x1)) / x1 - 2 * x1
        
        fcn = funAD.function(fcn_R1)
        assert fcn.x_dim ==1 and len(fcn.function_list)==1 and type(fcn.function_list)==tuple

        # R2
        def fcn_f_R2(x1,x2):
            return funAD.sin(funAD.exp(x1))
        def fcn_g_R2(x1,x2):
            return funAD.sin(funAD.exp(x1))
        fcn2 = funAD.function(fcn_f_R2,fcn_g_R2,x_dim=2)
        assert fcn2.x_dim ==2 and len(fcn2.function_list)==2 and type(fcn2.function_list)==tuple

    def test_grad(self):
        
        # check for value error (input dimension mismatch)
        # calculation accuracy tests
        def fcn_R1(x1):
            return funAD.sin(funAD.exp(x1)) + 6 * (funAD.cos(x1)) / x1 - 2 * x1
        fcn = funAD.function(fcn_R1)
        with pytest.raises(ValueError):
            fcn.grad([1,2])
        with pytest.raises(ValueError):
            fcn.grad(1,2)
    
        def fcn_f_R2(x1,x2):
            return funAD.sin(funAD.exp(x1))
        def fcn_g_R2(x1,x2):
            return funAD.sin(funAD.exp(x1))
        fcn2 = funAD.function(fcn_f_R2,fcn_g_R2,x_dim=2)
        with pytest.raises(ValueError):
            fcn2.grad(1)
        with pytest.raises(ValueError):
            fcn.grad([1,2],3)

        f = funAD.function(lambda x :x[0]+funAD.sin(x[1]), lambda x: x[0]*funAD.exp(x[1]),x_dim=2)
        assert f([2,3])[0] == 2.1411200080598674
        assert f([2,3])[1] == 40.171073846375336
        assert f.grad([2,3]) [0][0] == 1
        
        # correct x dimension input
        f2 = funAD.function(lambda x :x[0]*x[1]*x[2]*x[3]*x[4],x_dim=5)
        assert f2([2,1,1,1,1]) == 2

        # wrong x dimension input
        with pytest.raises(ValueError):
            f2.grad([1,2]) 
        with pytest.raises(ValueError):
            f2.grad([2,1,1,1,1],1)
        
        #R1 to R1
        def fcn_R1(x1):
            return funAD.sin(funAD.exp(x1)) + 6 * (funAD.cos(x1)) / x1 - 2 * x1
        fcn = funAD.function(fcn_R1)
        assert round(fcn(1),4) == 1.6526 and round(fcn.grad(1),6) == -12.768989
        
        # R2 to R1
        def fcn_R2(x1,x2):
            return funAD.sin(funAD.exp(x1)) + 6 * (funAD.cos(x2)) / x1 - 2 * x2
        fcn = funAD.function(fcn_R2,x_dim=2)
        assert round(fcn(1,1),6) == 1.652595 and round(fcn.grad(1,1)[0],6) == -5.720164 and round(fcn.grad(1,1)[1],6) == -7.048826
        
        # R2 to R2
        def fcn_f_R2(x1,x2):
            return funAD.sin(funAD.exp(x1))
        def fcn_g_R2(x1,x2):
            return 6 * (funAD.cos(x2)) / x1 - 2 * x2
        fcn = funAD.function(fcn_f_R2,fcn_g_R2,x_dim=2)
        assert [round(fcn(1,1)[0],6), round(fcn(1,1)[1],6)] == [0.410781, 1.241814]
        assert round(fcn.grad(1,1)[0][0],4) == -2.4783
        assert round(fcn.grad(1,1)[0][1],4) == -0.
        assert round(fcn.grad(1,1)[1][0],4) == -3.2418
        assert round(fcn.grad(1,1)[1][1],4) == -7.0488

        
