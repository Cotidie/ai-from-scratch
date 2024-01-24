import numpy as np
from math import sqrt
from enum import Enum
from .activation import Activation, Sigmoid

class InitMethod(Enum):
    GAUSSIAN = 1        # N(0, 1) 가우시안 분포로 초기화
    HE       = 2        # He Initialization (for ReLU)
    XAVIER   = 3        # Xavier Initialization (for sigmoid, tanh)

class Layer:
    """은닉층을 나타내는 클래스"""

    class Size:
        """레이어의 입력, 출력 크기를 나타내는 데이터 클래스"""
        def __init__(self, inputs: int, units: int):
            self.inputs = inputs
            self.units = units
            self.weights = None

    def __init__(self, inputs: int, units: int, activation: Activation=Sigmoid()) -> None:
        self.size = self.Size(inputs, units)
        self.activation = activation

    def init_parameters_gaussian(self) -> None:
        """모든 파라미터를 가우시안 분포 N(0, 1)로 초기화한다."""
        self.weights = np.random.randn(self.size.inputs)

    def init_parameters_he(self) -> None:
        """he 초기화 방식으로 파라미터를 초기화한다.

        입력의 개수 n에 대하여 각 파라미터를 가우시안 분포 N(0, 2/n)으로 초기화한다.
        """
        self.weights = np.random.randn(self.size.inputs)
        std = sqrt(2 / self.size.inputs)
        self.weights *= std

    def init_parameters_xavier(self) -> None:
        """Xavior 방식으로 파라미터를 초기화한다.

        입력의 개수 n에 대하여 각 파라미터를 균등 분포 U(-1/sqrt(n), 1/sqrt(n))으로 초기화한다.
        """
        bound = 1 / sqrt(self.size.inputs)
        lower, upper = -1 * bound, bound
        
        self.weights = np.random.rand(self.size.inputs)
        self.weights = lower + self.weights * (upper - lower)