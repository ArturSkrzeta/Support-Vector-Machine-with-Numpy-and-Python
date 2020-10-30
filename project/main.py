import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from svm import SVM

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
