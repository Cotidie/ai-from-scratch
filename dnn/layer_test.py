from .layer import Layer
from math import sqrt

class TestLayer:
    def test_he_initialization_should_follow_std_of_input_size(self):
        layer = Layer(100, 50)
        layer.init_parameters_he()
        
        expectedStd = sqrt(2 / layer.size.inputs)
        epsilon = 0.01
        std = layer.weights.std()
        
        assert layer.weights.shape == (layer.size.units, layer.size.inputs)
        assert (std < expectedStd+epsilon and std > expectedStd-epsilon) 