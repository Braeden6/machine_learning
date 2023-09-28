import numpy as np

'''
This class is used to normalize data and split it based off splits_percent.
Give X,Y and you will receive splits with train 80%, dev 10%, and test 10%, by default.
To change the splits, give a list of percentages that sum to 1. (train, dev, test, etc...)
'''
class Data:
    def __init__(self, X, Y, splits_percent=[0.8, 0.1, 0.1]):
        self.X = X
        self.Y = Y
        self.splits_percent = splits_percent
        self.check_data(self.X,self.Y,self.splits_percent)
        self.X_splits, self.Y_splits = self.split_data(self.X, self.Y, self.splits_percent)
        self.X_splits, self.mean, self.std = self.normalize_splits(self.X_splits)

    @staticmethod
    def check_data(X, Y, splits_percent):
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must have same number of rows")
        if len(splits_percent) < 1:
            raise Exception("splits_percent must be a list of at least one value, two for train and test, and three for train, dev, and test")
        if np.sum(splits_percent) != 1:
            raise Exception("splits_percent must sum to 1, otherwise data will be lost")

    @staticmethod
    def split_data(X, Y, splits_percent=[0.8, 0.1, 0.1]):
        '''
        Split data into sets based on the splits_percent.
        :param X: data
        :param Y: labels
        :param splits_percent: list of percentages of splits
        :return: list of tuples of (X,Y) splits
        '''
        Data.check_data(X,Y,splits_percent)
        m = X.shape[0]
        indexes = np.random.rand(m)
        max = splits_percent[0]
        min = 0
        X_splits = []
        Y_splits = []
        for i in range(len(splits_percent)):
            ith_indexes = (indexes < max) & (indexes >= min)
            X_splits.append(X[ith_indexes])
            Y_splits.append(Y[ith_indexes])
            if i != len(splits_percent) - 1:
                max += splits_percent[i+1]
                min += splits_percent[i]
        return X_splits, Y_splits

    @staticmethod
    def normalize_splits(X_splits):
        '''
        Normalize data using index zero as train to get mean and standard deviation.
        :param X_splits: list of X splits
        :return: list of normalized X splits, mean, and standard deviation
        '''
        if len(X_splits) < 1:
            raise Exception("No data to normalize")
        X_train = X_splits[0]

        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_splits = [(X - mean) / std for X in X_splits]
        return X_splits, mean, std
    
    @staticmethod
    def normalize(X, mean, std):
        '''
        Normalize data using given mean and standard deviation.
        '''
        return (X - mean) / std
    
    def get_train_data(self):
        return self.X_splits[0], self.Y_splits[0]
    
    def get_dev_data(self):
        if len(self.X_splits) < 2:
            raise Exception("No dev data, change splits_percent")
        return self.X_splits[1], self.Y_splits[1]
    
    def get_test_data(self):
        if len(self.X_splits) < 3:
            raise Exception("No test data, change splits_percent")
        return self.X_splits[2], self.Y_splits[2]


if __name__ == '__main__':
    length = 1000
    X = np.random.randn(length, 4)*100
    Y = np.random.randn(length, 1)
    data = Data(X, Y, [0.7,0.2,0.05,0.05])
    m = 0
    for i in range(len(data.X_splits)):
        # print(X_splits[i].shape, Y_splits[i].shape)
        m += data.X_splits[i].shape[0]
    # print(m, length)
    assert m == length

    print(data.X_splits[0][0:5])
    print(data.X_splits[1][0:5])

