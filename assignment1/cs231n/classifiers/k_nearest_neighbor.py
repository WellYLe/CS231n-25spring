from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练分类器 对于 k 近邻分类器来说，这只是忆训练数据。

        Inputs:
        - X: 形状为 (num_train, D) 的 numpy 数组，包含训练数据由每个维度为 D 的 num_train 样本组成。
        - y: 形状为 (N,) 的 numpy 数组，包含训练标签，其中y[i] 是 X[i] 的标签。
        """
        self.X_train = X 
        self.y_train = y
    #这部分都被写好了
    def predict(self, X, k=1, num_loops=0):
        """
        使用该分类器预测测试数据的标签。

        Inputs:
        - X: 形状为（num_test, D）的 numpy 数组，包含的测试数据有num_test个测试样本，每个样本的维数为D.
        - k: 为预测标签投票的近邻数量。
        - num_loops: 决定使用哪种实现来计算训练点和测试点之间的距离

        Returns:
        - y: 形状为（num_test,）的 numpy 数组，包含测试数据的预测标签，其中 y[i] 是测试点 X[i] 的预测标签。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        通过对训练数据和测试数据进行嵌套循环，计算X中每个测试点与self.X_train中每个训练点之间的距离。

        Inputs:
        - X: 一个形状为 (num_test, D) 的 numpy 数组，包含测试数据。

        Returns:
        - dists: 一个形状为(num_test, num_train)的numpy数组，其中dists[i, j]表示第i个测试点与第j个训练点之间的欧几里得距离。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the i_th test point and the j_th    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                
                diff = X[i] - self.X_train[j]      # 向量差
                dists[i, j] = np.sqrt(np.sum(diff ** 2))  # L2 距离pass
        return dists

    def compute_distances_one_loop(self, X):
        """
        计算X中每个测试点与self.X_train中每个训练点之间的距离，使用单次循环遍历测试数据。
        输入/输出：与compute_distances_two_loops相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            diff = self.X_train - X[i]       # 广播：训练集每一行 - X[i]
            dists[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        """
        计算X中每个测试点与self.X_train中每个训练点之间的距离，
        且不使用显式循环。

        输入/输出：与compute_distances_two_loops相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        X_square = np.sum(X**2, axis=1, keepdims=True)          # (num_test, 1)
        X_train_square = np.sum(self.X_train**2, axis=1)        # (num_train,)
        cross_term = X.dot(self.X_train.T)                      # (num_test, num_train)
        dists = np.sqrt(X_square + X_train_square - 2 * cross_term)
        return dists

    def predict_labels(self, dists, k=1):
        """
        给定测试点与训练点之间的距离矩阵，
        为每个测试点预测一个标签。

        输入：
        - dists：一个形状为 (num_test, num_train) 的 numpy 数组，其中 dists[i, j]
          表示第 i 个测试点与第 j 个训练点之间的距离。

        返回值：
        - y：形状为 (num_test,) 的 numpy 数组，包含测试数据的预测标签，其中 y[i] 是测试点 X[i] 的预测标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################


            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
        #        找出前 k 个最近邻
            nearest = np.argsort(dists[i])[:k]       # 按距离升序排序，取前 k
            closest_y = self.y_train[nearest]        # 这些点的标签
        # 投票（出现最多的标签）
            counts = np.bincount(closest_y)          # 统计频率
            y_pred[i] = np.argmax(counts)            # 选最多的那个

        return y_pred
