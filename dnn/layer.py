import numpy as np
from math import sqrt
from enum import Enum
from .activation import Activation, Sigmoid
from .loss_function import LossFunction

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

    class Cache:
        """Forward, Backward Pass 도중 계산된 값을 보관한다."""
        def __init__(self) -> None:
            self.pre_activation = None
            self.activated = None
            self.error_signal = None

    def __init__(self, inputs: int, units: int, activation: Activation=Sigmoid()) -> None:
        self.size = self.Size(inputs, units)
        self.activation = activation
        self.weights = None
        self.cache = self.Cache()


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

    def forward_pass(self, input: np.ndarray):
        """1개 데이터 샘플에 대해 선형 변환, 활성 함수를 계산한다. 
        
        입력수 i, 유닛 수 u에 대해 선형변환 a_u, 활성함수 z_u를 레이어에 저장한다.

        Args:
            input (np.ndarray): 1개 데이터 샘플의 입력 벡터 (i x 1)
        """
        linear_transform = np.dot(self.weights, input)      # u x 1
        activation = self.activation.calc(linear_transform) # u x 1

        self.cache.pre_activation = linear_transform 
        self.cache.activated = activation

        return activation

    def backpropagation(
            self, 
            next_weights: np.ndarray, 
            next_grads: np.ndarray
        ) -> np.ndarray:
        """에러 함수에 대해 현재 레이어의 pre-activation의 그라디언트를 계산한다. 이는 Chain Rule을 따른다.
            출력의 개수 o, 유닛의 개수 u에 대하여 현재 레이어의 에러 시그널 (u x 1)을 self.cache에 캐싱한다.
        Args:
            next_weights (np.ndarray): 다음 레이어가 가진 파라미터 (o x u)
            next_grads (np.ndarray): 다음 레이어의 에러 시그널 (o x 1)
        """

        error_signal = np.dot(
            np.transpose(next_weights),  # z x u
            next_grads                   # u x 1
        )                                # z x 1
        error_signal *= self.activation.derivative(self.cache.pre_activation)
        self.cache.error_signal = error_signal

    def create_error_signal(self, loss_function: LossFunction, y_train: np.ndarray):
        self.cache.error_signal = loss_function.derivative(
            self.cache.pre_activation,
            self.activation,
            y_train
        )
