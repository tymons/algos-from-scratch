import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn.knn import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
plt.scatter(X_train[:, 3], X_train[:, 2], c=y_train, cmap=cmap, edgecolor='k')

clf = KNN(4)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

plt.scatter(X_test[:, 3], X_test[:, 2], c=predictions, cmap=cmap, s=150, marker='x')
plt.show()