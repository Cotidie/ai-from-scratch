import numpy as np
from .activation import Activation

class LossFunction:
    def calc(self, predict: np.ndarray, truth: np.ndarray) -> float:
        raise NotImplementedError("Calc 함수를 구현해야 합니다.")
    
    def derivative(
            self, 
            pre_activation: np.ndarray,
            activation: Activation,
            truth: np.ndarray
        ) -> np.ndarray:
        raise NotImplementedError("derivative 함수를 구현해야 합니다.")


class CrossEntropy(LossFunction):
    def calc(self, predict: np.ndarray, truth: np.ndarray) -> float:
        """1개 예측에 대한 Loss 함수를 계산한다. SDG를 진행할 경우 이를 더하여 평균을 구해야 한다.
        
        Args:
            predict (np.ndarray): 모델이 예측한 값 (One-hot)
            truth (np.ndarray): 실제 값 (One-hot)

        Returns:
            float: Loss 함수의 값
        """
        positives = -np.dot(truth, np.log(predict))
        negatives = -np.dot(1-truth, np.log(1-predict))
        return positives + negatives


    def derivative(
            self, 
            pre_activation: np.ndarray,
            activation: Activation,
            truth: np.ndarray
        ) -> np.ndarray:
        """

        Args:
            pre_activation (np.ndarray): 출력 레이어의 pre-activation 항 (k x 1)
            activation (Activation): 출력 레이어의 activation 함수
            truth (np.ndarray): 실제 값

        Returns:
            np.ndarray: 각각의 유닛에 대한 그라디언트 값 (k x 1)
        """
        derivatives = activation.derivative(pre_activation) * truth
        derivatives /= activation.calc(pre_activation)

        return -derivatives
