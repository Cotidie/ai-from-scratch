from typing import List
from numpy import ndarray
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


        

    def forward_pass(self, x_sample: ndarray) -> ndarray:
        """1개 데이터 샘플에 대해 결과값을 계산한다.

        Args:
            x_sample (ndarray): 훈련 데이터의 입력 벡터

        Returns:
            ndarray: _description_
        """
        for layer in self.layers:
            (pre_activation, activation) = layer.forward_pass()


    def backpropagation(self):
        pass

    def test_accuracy(self):
        pass