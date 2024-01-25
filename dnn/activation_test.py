from .activation import Activation, Sigmoid, Identity, Softmax
import numpy as np

class TestActivation:
    def test_child_classes_should_inherit_activation_class(self):
        vector = np.random.rand(100)
        identity = Identity()
        sigmoid = Sigmoid()

        identity.calc(vector); identity.derivative(vector)
        sigmoid.calc(vector); sigmoid.derivative(vector) 

    def test_the_index_of_the_maximum_should_be_the_same_after_softmax(self):
        softmax = Softmax()
        x = np.array([1, 3, 2, 100, 2, 3, 2])
        max_index = 3

        activated = softmax.calc(x)
        assert activated.argmax() == max_index
        assert activated.size == x.size