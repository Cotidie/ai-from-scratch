from typing import List
from numpy import ndarray
from .layer import Layer

# Deep Neural Network를 나타내는 클래스
class Network:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

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
        

    def forward_pass(self):
        pass

    def backpropagation(self):
        pass

    def test_accuracy(self):
        pass