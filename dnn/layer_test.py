from .layer import Layer
from math import sqrt
import numpy as np

class TestLayer:
    def test_he_initialization_should_follow_std_of_input_size(self):
        layer = Layer(100, 50)
        layer.init_parameters_he()
        
        expectedStd = sqrt(2 / layer.size.inputs)
        epsilon = 0.01
        std = layer.weights.std()
        
        assert layer.weights.shape == (layer.size.units, layer.size.inputs)
        assert (std < expectedStd+epsilon and std > expectedStd-epsilon) 

    def test_forward_pass_should_return_arrays_of_size_of_units(self):
        unit_size = 50
        input_size = 30
        layer = Layer(inputs=input_size, units=unit_size)
        layer.init_parameters_gaussian()
        
        sample = np.random.rand(input_size)
        A, Z = layer.forward_pass(sample)

        assert A.size == unit_size
        assert Z.size == unit_size