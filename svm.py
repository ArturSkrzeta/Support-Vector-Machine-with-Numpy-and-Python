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

def visualize_svm(clf, x_sample, y_predicted):

     def get_hyperplane_value(x, w, b, offset):
          return (-w[0] * x + b + offset) / w[1]

     fig = plt.figure()
     ax = fig.add_subplot(1,1,1)

     # plotting data
     plt.scatter(X[:,0], X[:,1], marker='o')
     plt.scatter(x_sample[:,0], x_sample[:,1], marker='o')

     # setting separation lines
     x0_1 = np.amin(X[:,0]) # : - all from 0 column ==> min from first column
     x0_2 = np.amax(X[:,0]) #                       ==> max from first column

     x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
     x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

     x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
     x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

     x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
     x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

     # plotting separation gap
     ax.plot([x0_1, x0_2],[x1_1, x1_2], 'k')
     ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
     ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')

     # setting axis ranges up
     x1_min = np.amin(X[:,1])
     x1_max = np.amax(X[:,1])
     ax.set_ylim([x1_min-3,x1_max+3])

     plt.show()

def print_x_y(x, y):
    print('[feat1, feat2], label')
    for pair in zip(x,y):
        print(pair)

def main():

    new_sample = [[9.012, -1.1],[-4.01, -6.13],[4.12,-8.001],[0.1,-8.9822]]
    np_new_sample = np.array(new_sample)

    global X, y
    X, y =  datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)

    # print(X.shape)        =>  (50, 2) => features
    # print(y.shape)        =>  (50, )  => labels
    # print(np.zeros(2))    =>  [0. 0.]
    # print(X)              =>  [[  7.12731332  -4.4394424 ] [  6.68873898  -2.44840134] ... ]
    # print(y)              =>  [1 1 0 1 0 ... ]

    clf = SVM()
    clf.fit(X, y)

    predicted_labels_for_new_data = clf.predict(np_new_sample)

    print_x_y(new_sample, predicted_labels_for_new_data)

    visualize_svm(clf, np_new_sample, predicted_labels_for_new_data)

if __name__ == '__main__':
    main()
