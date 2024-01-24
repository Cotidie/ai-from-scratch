from keras.datasets import mnist
import matplotlib.pyplot as plt
from dnn import Layer, Network
        
if __name__ == "__main__":
    # x_train: 손글씨 이미지(28x28), 60,000개
    # y_train: 0~9 클래스값, 60,000개
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Deep Neural Network (3개 레이어)
    network = Network(
        layers=[
            Layer(28*28, 128),  # 은닉층 1: 784(28*28) 입력, 128개 유닛
            Layer(128, 64),     # 은닉층 2: 128개 입력, 64개 유닛
            Layer(64, 10)       # 출력: 64개 입력, 10개 출력 
        ]
    )
    print(type(x_train))
    # network.train(epoch=10, learnRate=0.001)
