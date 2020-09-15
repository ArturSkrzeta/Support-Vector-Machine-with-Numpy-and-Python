import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape # n_samples = 50   n_features = 2

        # [1 1 0 1 0 ... ] ==> [1 1 -1 1 -1 ... ]
        _y_train = np.where(y_train <= 0, -1, 1)

        self.w = np.zeros(n_features) # [0. 0.] as n_features = 2
        self.b = 0

        for _ in range(self.n_iters):
            for index, x_i in enumerate(X_train):
                condition = _y_train[index] * (np.dot(x_i, self.w) - self.b) >= 1
                print(_y_train[index] * (np.dot(x_i, self.w) - self.b))
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, _y_train[index]))
                    self.b -= self.lr * _y_train[index]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx) # np.sign([-5., 4.5]) ==> array([-1.,  1.])

def visualize_svm():

     def get_hyperplane_value(x, w, b, offset):
          return (-w[0] * x + b + offset) / w[1]

     fig = plt.figure()
     ax = fig.add_subplot(1,1,1)
     plt.scatter(X[:,0], X[:,1], marker='o',c=y)

     x0_1 = np.amin(X[:,0])
     x0_2 = np.amax(X[:,0])

     x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
     x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

     x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
     x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

     x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
     x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

     ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')
     ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
     ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')

     x1_min = np.amin(X[:,1])
     x1_max = np.amax(X[:,1])
     ax.set_ylim([x1_min-3,x1_max+3])

     plt.show()

def main():

    X, y =  datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)

    # print(X.shape)        =>  (50, 2) => features
    # print(y.shape)        =>  (50, )  => labels
    # print(np.zeros(2))    =>  [0. 0.]

    clf = SVM()
    clf.fit(X, y)
    # #predictions = clf.predict(X)
    #
    # print(clf.w, clf.b)

    # visualize_svm()

if __name__ == '__main__':
    main()
