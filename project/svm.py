import numpy as np

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X_train, y_train):

        #   X_train
        #   [[  7.12731332  -4.4394424 ]
        #    [  6.68873898  -2.44840134]
        #    [ -1.1004791   -7.78436803]
        #    [  3.99337867  -4.90451269]
        #    [ ... ... ]]

        n_samples, n_features = X_train.shape # n_samples = 50   n_features = 2

        # [1 1 0 1 0 ... ] ==> [1 1 -1 1 -1 ... ]
        _y_train = np.where(y_train <= 0, -1, 1)

        self.w = np.zeros(n_features) # [0. 0.] as n_features = 2
        self.b = 0

        for _ in range(self.n_iters):
            for index, x_i in enumerate(X_train):
                                #      (2,)                     *   (2,)
                                #   [[  7.12731332  -4.4394424 ] * [0. 0.]  = scalar value
                                #    [... ...]]
                condition = _y_train[index] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, _y_train[index]))
                    self.b -= self.lr * _y_train[index]

    def predict(self, X):
        # giving the labels: 1 or  -1 for the new comming in data
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx) # np.sign([-5., 4.5]) ==> array([-1.,  1.])
