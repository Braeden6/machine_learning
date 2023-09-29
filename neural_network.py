import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from utils import Data, one_hot_encode


class NeuralNetwork:
    def __init__(self, X, Y, iterations=1000, learning_rate=0.1, hidden_layers=[4], verbose=False):
        if len(Y.shape) == 1:
            raise Exception('Y must be a 2D array')
        X = X.T
        Y = Y.T

        self.init_weights(X, Y, hidden_layers)
        for i in range(iterations):
            A = self.forward_propagation(X)
            self.backward_propagation(Y)
            self.update_weights(learning_rate)
            cost = self.calculate_cost(A, Y, X)

            if verbose and i % 1000 == 0:
                print(f'iteration {i} cost is: {cost}')
    
    def forward_propagation(self, X):
        A = X
        Zs = []
        As = [X]

        for W, b in zip(self.W, self.b):
            Z = np.dot(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            
            Zs.append(Z)
            As.append(A)
        self.Zs = Zs
        self.As = As
        return As[-1]

    def predict(self, X):
        X = X.T
        A = self.forward_propagation(X)
        if A.shape[0] == 1:
            return A > 0.5
        return np.argmax(A, axis=0)

    def backward_propagation(self, Y):
        m = self.As[0].shape[1]
        L = len(self.W)
        
        dWs = []
        dbs = []

        # Start with the output layer
        dZ = self.As[-1] - Y
        for l in reversed(range(L)):
            dW = 1 / m * np.dot(dZ, self.As[l].T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dZ = np.dot(self.W[l].T, dZ) * self.As[l] * (1 - self.As[l])
            
            dWs.insert(0, dW) 
            dbs.insert(0, db)
            

        self.dWs = dWs
        self.dbs = dbs

    def update_weights(self, learning_rate):
        L = len(self.W)
        for l in range(L):
            self.W[l] -= learning_rate * self.dWs[l]
            self.b[l] -= learning_rate * self.dbs[l]

    def calculate_cost(self, A, Y, X):
        logProbs = np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), 1 - Y)
        return -1 / X.shape[1] * np.sum(logProbs)

    def init_weights(self, X, Y, hidden_layers):
        self.W = []
        self.b = []
        
        input_size = X.shape[0]
        
        # Initialize weights and biases for hidden layers
        for nodes in hidden_layers:
            W_layer = np.random.randn(nodes, input_size) * 0.01
            b_layer = np.zeros((nodes, 1))
            self.W.append(W_layer)
            self.b.append(b_layer)
            input_size = nodes

        # Initialize weights and biases for the output layer
        W_out = np.random.randn(Y.shape[0], input_size) * 0.01
        b_out = np.zeros((Y.shape[0], 1))
        
        self.W.append(W_out)
        self.b.append(b_out)

    def error(self, X, Y):
        Y = Y.T
        Y_prediction = self.predict(X)
        return np.mean(np.abs(Y_prediction - Y))


def show_digit(image):
    plt.imshow(image, cmap='gray')  # Use grayscale colormap
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

if __name__ == '__main__':
    # iris = datasets.load_iris()
    # X = iris["data"]
    # Y = (iris["target"] == 0).astype(np.int16).reshape((len(iris["target"]), 1))
    # data = Data(X, Y, [0.8,0.2])
    # X_train, y_train = data.get_train_data()
    # X_test, y_test = data.get_dev_data()
    
    # model = NeuralNetwork(X_train, y_train, 4_000, 0.01, [4], True)
    # print(f'error {model.error(X_test, y_test)}')
    digits = datasets.load_digits()
    X = digits["data"]
    print(X.shape)
    Y = one_hot_encode(digits["target"])
    data = Data(X, Y, [0.8,0.2])
    X_train, y_train = data.get_train_data()
    X_test, y_test = data.get_dev_data()


    model = NeuralNetwork(X_train, y_train, 5_000, 0.1, [], True)
    prediction = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    print(f'Acuraccy: {np.mean(prediction == y_test)}')


    for i in range(5):
        X_new = ((X[i] - data.mean)/data.std).reshape((1,-1))
        prediction = model.predict(X_new)
        print(f'prediction: {prediction}')
        show_digit(digits.images[i])








