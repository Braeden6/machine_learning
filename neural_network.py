import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import Data


class NeuralNetwork:
    def __init__(self, X, Y, iterations=1000, learning_rate=0.1, hidden_layers=[4], verbose=False):
        X = X.T
        Y = Y.reshape((1,len(Y)))

        self.init_weights(X, Y, hidden_layers)
        for i in range(iterations):
            A = self.forward_propagation(X)
            self.backward_propagation(Y)
            self.update_weights(learning_rate)
            cost = self.calculate_cost(A, Y, X)

            if verbose and i % 1000 == 0:
                print(f'cost: {cost}')
    
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
        return A > 0.5

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
        Y = Y.reshape((1,len(Y)))
        Y_prediction = self.predict(X)
        return np.mean(np.abs(Y_prediction - Y))


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris["data"]
    Y = (iris["target"] == 0).astype(np.int16)
    data = Data(X, Y, [0.8,0.2])
    X_train, y_train = data.get_train_data()
    X_test, y_test = data.get_dev_data()
    


    model = NeuralNetwork(X_train, y_train, 20_000, 0.1, [5,4], True)
    print(f'error {model.error(X_test, y_test)}')









