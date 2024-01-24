from .activation import Activation, Sigmoid, Identity
import numpy as np

class TestActivation:
    def test_child_classes_should_inherit_activation_class(self):
        vector = np.random.rand(100)
        identity = Identity()
        sigmoid = Sigmoid()

        identity.calc(vector); identity.derivative(vector)
        sigmoid.calc(vector); sigmoid.derivative(vector) 