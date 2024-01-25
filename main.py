from keras.datasets import mnist
from dnn import Layer, Network, InitMethod, Softmax, Sigmoid
import numpy as np

def convert_to_one_hot(numbers: np.ndarray) -> np.ndarray:
    converted = np.zeros((numbers.size, 10))
    converted[np.arange(numbers.size), numbers] = 1
    return converted

if __name__ == "__main__":
    # x: 손글씨 이미지(28x28)
    # y: 0~9 클래스값
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = convert_to_one_hot(y_train)
    y_test = convert_to_one_hot(y_test)

    network = Network(
        layers=[
            Layer(28*28, 128, activation=Sigmoid()),  # 은닉층 1: 784개 입력, 128개 출력
            Layer(128, 64, activation=Sigmoid()),     # 은닉층 2: 128개 입력, 64개 출력
            Layer(64, 10, activation=Softmax())       # 출력층:    64개 입력, 10개 출력 
        ]
    )
    network.initialize(InitMethod.XAVIER)
    network.train(x_train, y_train, epoch=10, learnRate=0.001)
    network.test_accuracy(x_test, y_test)
