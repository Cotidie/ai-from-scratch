import numpy as np
from math import sqrt
from enum import Enum
from .activation import Activation, Sigmoid

# TODO: 파라미터 초기화 클래스 분리

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

    def __init__(self, inputs: int, units: int, activation: Activation=Sigmoid()) -> None:
        self.size = self.Size(inputs, units)
        self.activation = activation
        self.weights = None

    def init_parameters_gaussian(self) -> None:
        """모든 파라미터를 가우시안 분포 N(0, 1)로 초기화한다."""
        self.weights = np.random.randn(self.size.units, self.size.inputs)

    def init_parameters_he(self) -> None:
        """he 초기화 방식으로 파라미터를 초기화한다.

        입력의 개수 n에 대하여 각 파라미터를 가우시안 분포 N(0, 2/n)으로 초기화한다.
        """
        std = sqrt(2 / self.size.inputs)

        self.weights = np.random.randn(self.size.units, self.size.inputs)
        self.weights *= std
        
    def init_parameters_xavier(self) -> None:
        """Xavior 방식으로 파라미터를 초기화한다.

        입력의 개수 n에 대하여 각 파라미터를 균등 분포 U(-1/sqrt(n), 1/sqrt(n))으로 초기화한다.
        """
        bound = 1 / sqrt(self.size.inputs)
        lower, upper = -1 * bound, bound

        self.weights = np.random.rand(self.size.units, self.size.inputs)
        self.weights = lower + self.weights * (upper - lower)

    def forward_pass(self, input: np.ndarray) -> (np.ndarray, np.ndarray):
        """1개 데이터 샘플에 대해 선형 변환, 활성 함수를 계산한다. 입력수 i, 유닛 수 u에 대해

        Args:
            input (np.ndarray): 1개 데이터 샘플의 입력 벡터 (i x 1)

        Returns:
            (np.ndarray, np.ndarray): 선형변환 a_u, 활성함수 z_u
        """
        linear_transform = np.dot(self.weights, input)      # u x 1
        activation = self.activation.calc(linear_transform) # u x 1

        return linear_transform, activation

    def backpropagation(self):
        pass

