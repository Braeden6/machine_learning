import numpy as np
import time
from sklearn import datasets
import matplotlib.pyplot as plt
from utils import Data, one_hot_encode

# TODO:
# https://chat.openai.com/share/bc7547c5-a987-4820-9fc7-78df16a2f39c
# - Gradient checking
# - Add regression support, MSE/MAE cost function

# Others:
# Regularization: L1, Elastic Net, Weight Noise, Weight Decay, 
#       Weight Constraints, Gradient Clipping, Max-Norm Regularization, Noise Injection
# NN Structure: ResNet, DenseNet ... etc
# Normalization: Min-Max scaling, Z-score normalization, Decimal Scaling, Mean Normalization, Unit Vector Scaling, Robust Scaling, Power Transformation, Log Transformation, 
#       Quantile Transformation, Batch Normalization, Layer Normalization, Contrast Normalization, Softmax Normalization 
# Activations: Leaky ReLU, Parametric ReLU, Swish, Softmax, ELU, SELU
# Batch normalization, Early stopping, Data augmentation
# Outlier detection  ...

# Actions:
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

# Learning Rate Decay:
def no_decay(learning_rate, epochs, iteration):
    return learning_rate

def exponential_decay(learning_rate, epochs, iteration, k=0.25):
    return learning_rate * np.exp(-iteration / (epochs * k))

def inverse_time_decay(learning_rate, epochs, iteration, k=40):
    return learning_rate / (1 + k * iteration/epochs)

# Linear by default
def polynomial_decay(learning_rate, epochs, iteration, k=1.02, p=1.0):
    return  learning_rate * (1 - iteration/(epochs*k))**p

class NeuralNetwork:
    def __init__(self, X, Y, 
                 epochs=1000, 
                 learning_rate=0.1, 
                 learning_rate_decay=polynomial_decay,
                 hidden_layers=[(4, Sigmoid)], 
                 output_layer=Sigmoid, 
                 lambd=0,  # 0 turns off L2 Regularization
                 keep_prob=1, # 1 turns off dropout 
                 mini_batch_size=0, # zero turns off mini batch
                 verbose=False):
        if len(Y.shape) == 1:
            raise Exception('Y must be a 2D array')
        X = X.T
        Y = Y.T
        if mini_batch_size == 0:
            mini_batch_size = X.shape[0]
        self.mini_batch_size = mini_batch_size
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.print_frequency = max(1, 10**(int(np.log10(self.epochs)) - 1))

        self.init_weights(X, Y, hidden_layers, output_layer)
        tic = time.time()
        for i in range(epochs):
            self.learning_rate = learning_rate_decay(learning_rate, epochs, i)
            
            # Create mini-batches
            mini_batch_X, mini_batch_Y = self.random_mini_batches(X, Y, self.mini_batch_size)
            
            for mini_batch_X, mini_batch_Y in zip(mini_batch_X, mini_batch_Y):
                A = self.forward_propagation(mini_batch_X)
                self.backward_propagation(mini_batch_Y)
                self.update_weights()
            
            # Calculate cost on the entire dataset or on a mini-batch for logging
            A = self.forward_propagation(X)
            cost = self.calculate_cost(A, Y, X)

            if verbose and i % self.print_frequency == 0:
                print(f'''epoch {i}:
                cost is: {cost} 
                took: {round(time.time() - tic, 3)} seconds
                learning rate: {self.learning_rate}''')
                tic = time.time()
    
    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=None):
        if seed:
            np.random.seed(seed)
        
        m = X.shape[1] 
        mini_batches_X = []
        mini_batches_Y = []
        
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        def get_batch(min,max, X,Y):
            return X[:,min:max], Y[:,min:max]

        num_complete_minibatches = m // mini_batch_size
        for k in range(0, num_complete_minibatches):
            mini_batch_X, mini_batch_Y = get_batch(k * mini_batch_size, (k + 1) * mini_batch_size, shuffled_X, shuffled_Y)
            mini_batches_X.append(mini_batch_X)
            mini_batches_Y.append(mini_batch_Y)

        if m % mini_batch_size != 0:
            mini_batch_X, mini_batch_Y = get_batch(num_complete_minibatches * mini_batch_size, m, shuffled_X, shuffled_Y)
            mini_batches_X.append(mini_batch_X)
            mini_batches_Y.append(mini_batch_Y)

        return mini_batches_X, mini_batches_Y

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
        self.dWs = []
        self.dbs = []
        # Start with the output layer
        dZ = self.As[-1] - Y
        for l in reversed(range(L)):
            dropout = np.random.rand(self.As[l].shape[0], self.As[l].shape[1]) < self.keep_prob
            A = self.As[l] * dropout
            A /= self.keep_prob
            dW = 1 / m * np.dot(dZ, A.T) + (self.lambd / m) * self.W[l]
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dZ = np.dot(self.W[l].T, dZ) * self.activation_functions[l].derivative(A)
            self.dWs.insert(0, dW) 
            self.dbs.insert(0, db)

    def predict(self, X):
        X = X.T
        A = self.forward_propagation(X)
        if A.shape[0] == 1:
            return A > 0.5
        return np.argmax(A, axis=0)

    def update_weights(self):
        L = len(self.W)
        for l in range(L):
            self.W[l] -= self.learning_rate * self.dWs[l]
            self.b[l] -= self.learning_rate * self.dbs[l]

    def calculate_cost(self, A, Y, X, epsilon=1e-15):
        A = np.clip(A, epsilon, 1 - epsilon)
        logProbs = np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), 1 - Y)
        cross_entropy_cost = -1 / X.shape[1] * np.sum(logProbs)

        # if L2 regularization is used
        if self.lambd > 0:
            m = X.shape[1]
            L2_cost = 0
            for l in range(len(self.W)):
                L2_cost += np.sum(np.square(self.W[l]))
            L2_cost *= self.lambd / (2 * m)
            return cross_entropy_cost + L2_cost
        
        return cross_entropy_cost

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

def plot_decay(learning_rate=0.1, epochs=10000):
    iterations = np.arange(0, epochs, 1)

    # Compute learning rates for all iterations
    inv_decay = [inverse_time_decay(learning_rate, epochs, it) for it in iterations]
    linear_decay = [polynomial_decay(learning_rate, epochs, it, p=1) for it in iterations]
    quadratic_decay = [polynomial_decay(learning_rate, epochs, it, p=2) for it in iterations]
    exp_decay = [exponential_decay(learning_rate, epochs, it) for it in iterations]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, inv_decay, label='Inverse Time Decay', color='blue')
    plt.plot(iterations, linear_decay, label='Linear Decay', color='red')
    plt.plot(iterations, quadratic_decay, label='Quadratic Decay', color='green')
    plt.plot(iterations, exp_decay, label='Exponential Decay', color='orange')
    plt.title('Learning Rate Scheduling: Inverse Time Decay')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    step = 'vis_model' # data, vis_model, vis_decay
    if step == 'data':
        print('Loading data...')
        data = datasets.load_digits()
        # data = datasets.load_iris()
        X = data["data"]
        Y = one_hot_encode(data["target"])

        data = Data(X, Y, [0.8,0.2])
        X_train, y_train = data.get_train_data()
        X_test, y_test = data.get_dev_data()

        model = NeuralNetwork(X_train, y_train, 
                              epochs=4000, 
                              learning_rate=lambda x: x*0.9999, 
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

    elif step == 'vis_model':
        # make_circles
        # make_classification
        X, Y = datasets.make_moons(n_samples=10000, noise=.2, random_state=0)
        Y = Y.reshape((-1,1))
        model = NeuralNetwork(X, Y, 
                              epochs=100, 
                              learning_rate=0.1, 
                              learning_rate_decay=polynomial_decay,
                              hidden_layers=[(10,Relu),(15,Relu), (15,Relu), (10,Relu)], 
                              output_layer=Relu, 
                              lambd=0.1,
                              keep_prob=0.8,
                              mini_batch_size=64,
                              verbose=True)
        prediction = model.predict(X=X)
        print(f'Accuracy: {np.mean(prediction == Y.T)}')

        plot_decision_boundary(model, X[0:200], Y[0:200])

    elif step == 'vis_decay':
        plot_decay(learning_rate=0.1, epochs=10000)
    







