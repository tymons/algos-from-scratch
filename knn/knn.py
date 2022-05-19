import numpy as np

from collections import Counter


def euclidian_dist(u, v):
    return np.sqrt(np.sum((u - v) ** 2))


class KNN:
    def __init__(self, neighbors: int = 3):
        self._k = neighbors

    def fit(self, X_train, y_train):
        self.train_data = X_train
        self.y_train = y_train

    def predict(self, X_test):
        return np.array([self._nearest(test_point) for test_point in X_test])

    def _nearest(self, X):
        """ Calculate euclidian distances for every point and find the k nearest"""
        distances = [euclidian_dist(X, train_point) for train_point in self.train_data]
        k_best_distances = self.y_train[np.argsort(distances)[:self._k]]
        predicted_class = Counter(k_best_distances).most_common(1)[0][0]
        return predicted_class

