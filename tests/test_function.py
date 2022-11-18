import funAD
import pytest


class TestTypes():
    def test_class_function(self):

        f = funAD.function(lambda x :x[0]+funAD.sin(x[1]), lambda x: x[0]*funAD.exp(x[1]),x_dim=2)

        assert f([2,3])[0] == 2.1411200080598674
        assert f([2,3])[1] == 40.171073846375336
        assert f.grad([2,3]) [0][0] == 1
        #assert f.grad([2,3]) [0][1] == -0.9899925 

    