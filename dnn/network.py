from typing import List
from numpy import ndarray, zeros
from .layer import Layer, InitMethod
from .loss_function import LossFunction

class Network:
    """Deep Neural Network를 나타내는 클래스"""
    def __init__(self, layers: List[Layer], loss: LossFunction) -> None:
        self.layers = layers
        self.loss = loss

    def initialize(self, method: InitMethod):
        """파라미터를 지정된 방식으로 초기화한다.

        Args:
            method (InitMethod): 파라미터 초기화 방식 (He, Gaussian 등)
        """
        if method == InitMethod.HE:
            for layer in self.layers:
                layer.init_parameters_he()
        if method == InitMethod.GAUSSIAN:
            for layer in self.layers:
                layer.init_parameters_gaussian()
        if method == InitMethod.XAVIER:
            for layer in self.layers:
                layer.init_parameters_xavier()

    def train(self,
              x_train: ndarray, 
              y_train: ndarray, 
              epoch: int=10, 
              learnRate: float=0.001
        ) -> None:
        """주어진 학습 데이터를 통해 모델의 파라미터를 학습한다.

        Args:
            x_train (ndarray): 학습 데이터 (입력)
            y_train (ndarray): 학습 데이터 (정답)
            epoch (int):       데이터셋 반복 학습 횟수 (기본 10)
            learnRate (float): 학습률 (기본 0.001)
        """

        for _ in range(epoch):
            for sample in range(x_train.shape[0]):
                # 1. 레이어별 pre-activation, output 값을 기록한다.
                input = x_train[sample]
                for layer in self.layers:
                    input = layer.forward_pass(input)
                
                # 2. backpropagation: 레이어별 에러 시그널을 기록한다.
                self.layers[-1].create_error_signal(
                    self.loss,
                    y_train[sample]
                )
                for i in range(len(self.layers)-2, 0, -1):
                    self.layers[i].backpropagation(
                        self.layers[i+1].weights,
                        self.layers[i+1].cache.error_signal
                    )

                # 3. 파라미터를 업데이트한다.
                # TODO: Vectorize


    def predict(self, sample):
        output = sample
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output

    def test_accuracy(self, x_test: ndarray, y_test: ndarray):
        correct = 0
        
