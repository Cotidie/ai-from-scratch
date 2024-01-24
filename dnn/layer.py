import numpy as np
from math import sqrt
from enum import Enum

class InitMethod(Enum):
    GAUSSIAN = 1
    HE       = 2

class Layer:
    """은닉층을 나타내는 클래스"""
    class Size:
        """레이어의 입력, 출력 크기를 나타내는 데이터 클래스"""
        def __init__(self, inputs: int, units: int):
            self.inputs = inputs
            self.units = units
            self.weights = None

    def __init__(self, inputs: int, units: int) -> None:
        self.size = self.Size(inputs, units)

    def init_parameters_he(self) -> None:
        """he 초기화 방식으로 파라미터를 초기화한다.

        입력의 개수 n에 대하여 각 파라미터를 가우시안 분포 N(0, 2/n)으로 초기화한다.
        """
        self.weights = np.random.randn(self.size.inputs)
        std = sqrt(2 / self.size.inputs)
        self.weights *= std