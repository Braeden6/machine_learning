import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
import numpy as np

def prepare_data(X, Y):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest, Ytrain, Ytest

class LogisticRegression:
    def __init__(self, X, Y, iterations=1000, learning_rate=0.01, verbose=False):
        self.X = X.T
        self.Y = Y
        self.w = np.zeros((X.shape[1], 1))
        self.b = 0
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.gradient_descent(iterations)

    def sigmoid(self, z):
        print(z.shape)
        s = 1 / (1 + np.exp(-z))
        return s
    
    def get_activation(self, X):
        return self.sigmoid(np.dot(self.w.T, X) + self.b)
    
    def get_cost(self, A):
        m = self.X.shape[1]
        return -1 / m * np.sum(self.Y * np.log(A) + (1 - self.Y) * np.log(1 - A))

    def predict(self, X):
        X = X.T
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        A = self.get_activation(X)
        for i in range(A.shape[1]):
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        return Y_prediction

    def gradient_descent(self, iterations):
        m = self.X.shape[1]
        for i in range(iterations):
            A = self.get_activation(self.X)
            cost = self.get_cost(A)
            # calculate gradients
            dw = 1 / m * np.dot(self.X, (A - self.Y).T)
            db = 1 / m * np.sum(A - self.Y)
            # update values
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            # self.learning_rate *= 0.999
            if self.verbose and i % 100 == 0:
                print("cost after iteration %i: %f" % (i, cost))

    def error(self, X, Y):
        Y_prediction = self.predict(X)
        return np.mean(np.abs(Y_prediction - Y))


# sigmoid: 0 to 1 = 1 / (1 + e^(-z)) || derivative: s * (1 - s) || g'(z) = a * (1 - a)
# tanh: -1 to 1 = (e^z - e^(-z)) / (e^z + e^(-z)) = tanh(z) || derivative: 1 - tanh(z)^2 || g'(z) = 1 - a^2
# relu: 0 to infinity = max(0, z) || derivative: 0 if z < 0 else 1 || g'(z) = 0 if z < 0 else 1
# leaky relu: 0 to infinity = max(0.01 * z, z) || derivative: 0.01 if z < 0 else 1 || g'(z) = 0.01 if z < 0 else 1


# why use linear activation function (only on output layer)
# example cost of house vs size output can be 0 to infinity





# HEART dataset
print('================= HEART DATASET =================')
data = pd.read_csv('heart.csv')
Y = data['output']
Y = Y.values
X = data.drop('output', axis=1).values
Xtrain, Xtest, Ytrain, Ytest = prepare_data(X, Y)

model = LogisticRegression(Xtrain, Ytrain, 500, 0.01, False)   
print(f'My error {round(model.error(Xtest, Ytest), 2)}')

clf = LR(solver='lbfgs', random_state=42)
clf.fit(Xtrain, Ytrain)
y_pred = clf.predict(Xtest)
accuracy = np.mean(y_pred == Ytest)
print(f'Sklearn Error:: {1 - accuracy}')

# IRIS dataset
print('================= IRIS DATASET =================')
iris = datasets.load_iris()
X = iris["data"]
y = (iris["target"] == 0).astype(np.int16)
X_train, X_test, y_train, y_test = prepare_data(X, y)


model = LogisticRegression(X_train, y_train, 4000, 0.01)   
print(f'My error {round(model.error(X_test, y_test), 2)}')

clf = LR(solver='lbfgs', random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Sklearn Error: {1 - accuracy}')
print('================= END =================')


