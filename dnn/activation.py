from numpy import ndarray, exp

class Activation:
    def calc(self, x: ndarray) -> ndarray:
        pass

    def derivative(self, x: ndarray) -> ndarray:
        pass
    
class Sigmoid(Activation):
    def __init__(self) -> None:
        pass

    def calc(self, x: ndarray):
        return 1 / (1+exp(-x))
    
    def derivative(self, x: ndarray):
        applied = self.calc(x)
        return applied * (1 - applied)