import funAD


class TestTypes():
    def test_class_function(self):

        def f(x):
            f1 = x[0]+funAD.sin(x[1])
            f2 = x[0]*funAD.exp(x[1])
            return [f1,f2]

        f = funAD.function(f)
        assert f([2,3])[0] == 2.1411200080598674
        assert f([2,3])[1] == 40.171073846375336
        assert f.grad([2,3]) [0][0] == 1
        #assert f.grad([2,3]) [0][1] == -0.9899925 

    