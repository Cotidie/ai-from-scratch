from numpy import ndarray, exp, ones, sum

class Activation:
    def calc(self, x: ndarray) -> ndarray:
        raise NotImplementedError("calc 함수를 구현해야 합니다.")

    def derivative(self, x: ndarray) -> ndarray:
        raise NotImplementedError("derivative 함수를 구현해야 합니다.")

class Sigmoid(Activation):
    def calc(self, x: ndarray) -> ndarray:
        return 1 / (1+exp(-x))
    
    def derivative(self, x: ndarray) -> ndarray:
        applied = self.calc(x)
        return applied * (1 - applied)

class Softmax(Activation):
    def calc(self, x: ndarray) -> ndarray:
        exps = exp(x-x.max())
        return exps / sum(exps)
    
    def derivative(self, x: ndarray) -> ndarray:
        applied = self.calc(x)
        return applied * (1-applied)
    
class Identity(Activation):
    def calc(self, x: ndarray) -> ndarray:
        return x
    
    def derivative(self, x: ndarray) -> ndarray:
        return ones(x.shape)