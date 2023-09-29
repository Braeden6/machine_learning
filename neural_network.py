import numpy as np
import time
from sklearn import datasets
import matplotlib.pyplot as plt
from utils import Data, one_hot_encode
from enum import Enum

# TODO:
# https://chat.openai.com/share/bc7547c5-a987-4820-9fc7-78df16a2f39c
# - Add more activation functions: tanh, relu. Other in future Leaky ReLU, Parametric ReLU, Swish, Softmax, ELU, SELU
# - Add regularization L2 and Dropout. Other things in future: L1 (maybe not as it isnt used much), 
#       Elastic Net, Weight Noise, Weight Decay, Weight Constraints, Gradient Clipping, Max-Norm Regularization, Noise Injection
# - Mini-batch gradient descent
# - Learning rate decay

# - Gradient checking


# - Others: Batch normalization, Early stopping, Data augmentation

# - ResNet, DenseNet ... etc

# - Min-Max scaling, Z-score normalization, Decimal Scaling, Mean Normalization, Unit Vector Scaling, Robust Scaling, Power Transformation, Log Transformation, 
#       Quantile Transformation, Batch Normalization, Layer Normalization, Contrast Normalization, Softmax Normalization 

# Outlier detection  ...

class Sigmoid:
    @staticmethod
    def function(Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def derivative(Z):
        s = Sigmoid.function(Z)
        return s * (1 - s)

    @staticmethod
    def init_weights(nodes, prev_nodes):
        return np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)

class Relu:
    @staticmethod
    def function(Z):
        return np.maximum(0, Z)

    @staticmethod
    def derivative(Z):
        return np.where(Z > 0, 1, 0)

    @staticmethod
    def init_weights(nodes, prev_nodes):
        return np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)

class Tanh:
    @staticmethod
    def function(Z):
        return np.tanh(Z)

    @staticmethod
    def derivative(Z):
        return 1.0 - np.tanh(Z)**2

    @staticmethod
    def init_weights(nodes, prev_nodes):
        # Xavier/Glorot initialization is often used for tanh
        return np.random.randn(nodes, prev_nodes) * np.sqrt(1. / prev_nodes)



class NeuralNetwork:
    def __init__(self, X, Y, iterations=1000, learning_rate=0.1, hidden_layers=[(4, Sigmoid)], output_layer=Sigmoid, verbose=False):
        if len(Y.shape) == 1:
            raise Exception('Y must be a 2D array')
        X = X.T
        Y = Y.T
        self.init_weights(X, Y, hidden_layers, output_layer)
        tic = time.time()
        for i in range(iterations):
            A = self.forward_propagation(X)
            self.backward_propagation(Y)
            self.update_weights(learning_rate)
            cost = self.calculate_cost(A, Y, X)

            if verbose and i % 1000 == 0:
                print(f'iteration {i} cost is: {cost} took: {round(time.time() - tic, 3)} seconds')
                tic = time.time()
    
    def forward_propagation(self, X):
        A = X
        Zs = []
        As = [X]

        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = np.dot(W, A) + b
            A = self.activation_functions[i].function(Z)
            
            Zs.append(Z)
            As.append(A)
        self.Zs = Zs
        self.As = As
        return As[-1]
    
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
            dZ = np.dot(self.W[l].T, dZ) * self.activation_functions[l].derivative(self.As[l])
            
            dWs.insert(0, dW) 
            dbs.insert(0, db)
            

        self.dWs = dWs
        self.dbs = dbs

    def predict(self, X):
        X = X.T
        A = self.forward_propagation(X)
        if A.shape[0] == 1:
            return A > 0.5
        return np.argmax(A, axis=0)

    def update_weights(self, learning_rate):
        L = len(self.W)
        for l in range(L):
            self.W[l] -= learning_rate * self.dWs[l]
            self.b[l] -= learning_rate * self.dbs[l]

    def calculate_cost(self, A, Y, X):
        logProbs = np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), 1 - Y)
        return -1 / X.shape[1] * np.sum(logProbs)

    def init_weights(self, X, Y, hidden_layers, output_layer):
        self.W = []
        self.b = []
        self.activation_functions = []
        input_size = X.shape[0]
        
        # Initialize weights and biases for hidden layers
        i= 0
        for nodes, activation_function in hidden_layers:
            W_layer = activation_function.init_weights(nodes, input_size)
            b_layer = np.zeros((nodes, 1))
            
            i += 1
            self.W.append(W_layer)
            self.b.append(b_layer)
            self.activation_functions.append(activation_function)
            input_size = nodes

        # Initialize weights and biases for the output layer
        W_out = np.random.randn(Y.shape[0], input_size) * 0.01
        b_out = np.zeros((Y.shape[0], 1))
        self.activation_functions.append(output_layer)
        self.W.append(W_out)
        self.b.append(b_out)

    def error(self, X, Y):
        Y = Y.T
        Y_prediction = self.predict(X)
        return np.mean(np.abs(Y_prediction - Y))


def show_digit(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def plot_decision_boundary(model, X, Y):
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.75)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == '__main__':
    visual_2D = True
    if not visual_2D:
        print('Loading data...')
        data = datasets.load_digits()
        # data = datasets.load_iris()
        X = data["data"]
        Y = one_hot_encode(data["target"])

        data = Data(X, Y, [0.8,0.2])
        X_train, y_train = data.get_train_data()
        X_test, y_test = data.get_dev_data()

        model = NeuralNetwork(X_train, y_train, 
                              iterations=4000, 
                              learning_rate=0.01, 
                              hidden_layers=[(4,Tanh)], 
                              output_layer=Sigmoid, 
                              verbose=True)
        prediction = model.predict(X_test)
        y_test = np.argmax(y_test, axis=1)
        print(f'Accuracy: {np.mean(prediction == y_test)}')

        # for i in range(5):
        #     X_new = ((X[i] - data.mean)/data.std).reshape((1,-1))
        #     prediction = model.predict(X_new)
        #     print(f'prediction: {prediction}')
        #     show_digit(data.images[i])

    else:

        X, Y = datasets.make_moons(n_samples=400, noise=0.2, random_state=0)
        Y = Y.reshape((-1,1))

        model = NeuralNetwork(X, Y, 
                              iterations=20000, 
                              learning_rate=0.1, 
                              hidden_layers=[(6,Tanh),(6,Relu),(6,Relu),(6,Relu),(6,Relu)], 
                              output_layer=Relu, 
                              verbose=True)
        prediction = model.predict(X=X)
        print(f'Accuracy: {np.mean(prediction == Y.T)}')

        plot_decision_boundary(model, X, Y)


    







