import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




class NeuralNetwork:
    def __init__(self, X, Y):
        self.m = X.shape[1]
        self.X = X
        self.Y = Y
        self.learning_rate = 0.01
        self.layers = [
            [
                np.random.randn(4, X.shape[0]) * 0.01,
                np.zeros((4, 1))
            ],
            [
                np.random.randn(Y.shape[0], 4) * 0.01,
                np.zeros((Y.shape[0], 1))
            ],
        ]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward_propagation(self, X):
        Z1 = np.dot(self.layers[0][0], X) + self.layers[0][1]
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.layers[1][0], A1) + self.layers[1][1]
        A2 = self.sigmoid(Z2)
        return A1, A2, Z1, Z2

    def backward_propagation(self, Y, X, A1, A2):
        dZ2 = A2 - Y
        dW2 = 1 / self.m * np.dot(dZ2, A1.T)
        db2 = 1 / self.m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.layers[1][0].T, dZ2) * A1 * (1 - A1)   #(1 - np.power(A1, 2))
        dW1 = 1 / self.m * np.dot(dZ1, X.T)
        db1 = 1 / self.m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2, dZ1, dZ2


    
    def get_cost(self, A, Y):
        logProbs = np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), 1 - Y)
        return -1 / self.m * np.sum(logProbs)

    def predict(self, X):
        A1, A2, Z1, Z2 = self.forward_propagation(X)
        predictions = A2 > 0.5
        return predictions
    
    def train(self, iterations):
        for i in range(iterations):
            A1, A2, Z1, Z2 = self.forward_propagation(self.X)
            dW1, db1, dW2, db2, dZ1, dZ2 = self.backward_propagation(self.Y, self.X, A1, A2)

            self.layers[0][0] -= self.learning_rate * dW1
            self.layers[0][1] -= self.learning_rate * db1
            self.layers[1][0] -= self.learning_rate * dW2
            self.layers[1][1] -= self.learning_rate * db2

            if i % 1000 == 0:
                cost = self.get_cost(A2, self.Y)
                print("cost after iteration %i: %f" % (i, cost))


        return 0

X = np.array([[1, 2, 3, 4], [2, 4, 1, 5]])  
Y = np.array([[0, 1, 0, 1]])
print(X.shape)

def prepare_data(X, Y):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest, Ytrain.reshape((len(Ytrain),1)), Ytest.reshape((len(Ytest),1))

iris = datasets.load_iris()
X = iris["data"]
y = (iris["target"] == 0).astype(np.int16)
X_train, X_test, y_train, y_test = prepare_data(X, y)

print(X_train.shape)

# model = NeuralNetwork(X_train, y_train)
# model.train(10000)

# predictions = model.predict(X_test)

# print("Accuracy: %f" % (np.sum(predictions == y_test) / len(y_test)))








