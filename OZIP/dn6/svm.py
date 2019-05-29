import numpy as np
from sklearn import datasets
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import glob
import gzip

class SVM:
    """
    Kernel based Support Vector Machine

    :param C: punishment for wrong classification
    :param rate: learn rate
    :param epoch: number of epochs
    :param coef_: lagrange coefficients
    :param X: learn data
    :param y: learn classes
    :param b: root
    :param kernel: kernel
    """
    def __init__(self, C=1, rate=0.001, epochs=5000, kernel="linear"):
        self.C = float(C)
        self.rate = rate
        self.epochs = epochs
        self.coef_ = None
        self.X = None
        self.y = None
        self.b = 0
        kernels = {'linear': self._linear, 'rbf': self._rbf, 'text': self._text}
        self.kernel = kernels[kernel]

    def fit(self, X, y):
        """
        Fit the SVM model

        :param X: X data
        :param y: y data
        :return: Lagrange coefficient for the learned model
        """
        self.coef_ = np.zeros(X.shape[0])
        self.X = X.copy()
        self.y = np.array([-1 if yi == 0 else 1 for yi in y])

        K = np.array([[self.kernel(self.X[i], self.X[j]) for j in range(X.shape[0])] for i in range(X.shape[0])])

        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                delta = self.rate * (1. - self.y[i] * np.sum(np.multiply(np.multiply(self.coef_, self.y), K[i])))
                self.coef_[i] = min(self.C, max(0., self.coef_[i] + delta))

        self.b = np.sum([self.y[i] - np.sum(self.coef_ * self.y * K[i])
                         for i in range(self.X.shape[0])]) / self.X.shape[0]
        return self.coef_

    def predict(self, X):
        """
        Predict classes for every row in  X.

        :param X: numpy array, data to classify
        :return: numpy array, classes for every row
        """
        y_pred = np.zeros(X.shape[0])
        for i in range(len(y_pred)):
            y_pred[i] = np.sum([self.coef_[j] * self.y[j] * self.kernel(self.X[j], X[i]) for j in range(self.X.shape[0])]) + self.b
        return np.array([0 if y < 0 else 1 for y in y_pred])

    def get_weights(self):
        """
        Return weights for a model.

        :return: numpy array, weights
        """
        return np.sum([self.coef_[n] * self.y[n] * self.X[n] for n in range(len(self.y))], axis=0)

    @staticmethod
    def _linear(X, x):
        """
        Linear kernel
        :param X: matrix or vector
        :param x: vector
        :return: similarity
        """
        return np.inner(X, x)

    @staticmethod
    def _rbf(X, x, gamma=.5):
        """
        Radial basis function kernel
        :param X: vector
        :param x: vector
        :param gamma: float, gamma
        :return: similarity
        """
        return np.exp(-gamma * np.linalg.norm(np.subtract(X, x)) ** 2)

    @staticmethod
    def _text(a, b):
        """
        Kernel for text data
        :param a: text a
        :param b: text b
        :return: similarity
        """
        a = str(a).encode('utf-8')
        b = str(b).encode('utf-8')
        la = len(gzip.compress(a))
        lb = len(gzip.compress(b))
        laa = len(gzip.compress(a+a))
        lbb = len(gzip.compress(b+b))
        lab = len(gzip.compress(a+b))
        lba = len(gzip.compress(b+a))

        delt_ab = lab - la
        delt_ba = lba - lb
        delt_aa = laa - la
        delt_bb = lbb - lb
        return  100-.5*((delt_ab - delt_bb) / delt_bb + (delt_ba - delt_aa) / delt_aa)

def add_ones(X):
    """
    Add ones to numpy array
    :param X: numpy array
    :return: numpy array
    """
    return np.column_stack((np.ones(len(X)), X))

def generate_data(data_type, n_samples=100):
    """
    Generate random data (taken from unit test)
    :param data_type:
    :param n_samples:
    :return:
    """
    np.random.seed(42)
    if data_type == "blobs":
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            centers=[[2, 2], [1, 1]],
            cluster_std=0.4
        )
    elif data_type == "circle":
        X = (np.random.rand(n_samples, 2) - 0.5) * 20
        y = (np.sqrt(np.sum(X ** 2, axis=1)) > 8).astype(int)
    else:
        return None
    X = add_ones(X)
    return X, y

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(add_ones(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def get_text_data(origin="text-data"):
    dirs = glob.glob(origin + "/*")
    X, y = [], []
    for i, d in enumerate(dirs):
        files = glob.glob(d + "/*")
        for file_name in files:
            with open(file_name, "rt", encoding="utf8") as file:
                X.append(" ".join(file.readlines()))
        y.extend([i] * len(files))
    return np.array(X), np.array(y)

if __name__ == '__main__':

    # text
    print("Classifying 'Homo Sapiens' and 'Salmonela Enterica' partial DNA sequences...")
    Xl, yl = get_text_data("text/data")
    Xt, yt = get_text_data("text/test")
    svm = SVM(C=1, rate=0.001, epochs=100, kernel="text")
    svm.fit(Xl, yl)
    predicted = svm.predict(Xt)
    print(yt)
    print(predicted)
    print("Precision: " + str(precision_score(yt, predicted)))

    # blobs
    print("Blobs dataset...")
    X, y = generate_data("blobs", n_samples=100)
    svm = SVM(C=1, rate=0.001, epochs=5000, kernel="linear")
    svm.fit(X, y)
    X0, X1 = X[:, 1], X[:, 2]
    xx, yy = make_meshgrid(X0, X1)


    plot_contours(plt, svm, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50 * svm.coef_, edgecolors='k')
    plt.savefig("blobs.pdf")
    plt.show()

    # circle
    print("Circle dataset...")
    X, y = generate_data("circle", n_samples=200)
    ind_pos = y > 0
    ind_neg = y < 1
    svm = SVM(C=1, rate=0.001, epochs=5000, kernel="rbf")
    svm.fit(X, y)

    X0, X1 = X[:, 1], X[:, 2]
    xx, yy = make_meshgrid(X0, X1)


    plot_contours(plt, svm, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50*svm.coef_, edgecolors='k')
    plt.savefig("circle.pdf")
    plt.show()