from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import datasets
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import normalize


class NeuralNetwork(MLPClassifier):
    """
    Feed forward neural network with one bias for each layer.
    :param hidden_layer_sizes: list of sizes of hidden layers.
    :param alpha: regularization rate.
    """

    def __init__(self, hidden_layer_sizes=(100,), alpha=0.0001):
        self.len_hidden_layers = len(hidden_layer_sizes)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.activation = self.sigmoid

        self.W = None
        self.n_classes = None
        self.X = None
        self.y = None
        self.W_mask = None

    def sigmoid(self, x):
        """
        Sigmoid function
        :param x: number/vector/matrix
        :return: sigmoid of x
        """
        return 1. / (1 + np.exp(-x))

    def set_data_(self, X, y):
        """
        Prepare model for training.
        :param X: data
        :param y: target
        """
        self.X = X
        self.n_classes = len(np.unique(y))

        self.y = y
        self.y_hot = np.array([list(1 if y[i] == x else 0 for x in range(self.n_classes)) for i in range(len(y))])
        self.n_input = self.X.shape[1]

        # prepare layers
        rows = self.X.shape[0]
        A = list()
        A.append(np.zeros((rows, self.n_input + 1)))

        for layer_size in self.hidden_layer_sizes:
            A.append(np.zeros((rows, layer_size + 1)))

        A.append(np.zeros((rows, self.n_classes)))
        self.A = A

    def init_weights_(self):
        """
        Initialize weights based on layer sizes.
        :return: vector of random coefficients.
        """
        sz = sum([(self.A[i].shape[1]) * (self.A[i + 1].shape[1] - 1) for i in range(len(self.A) - 2)]) + (
                    self.A[-2].shape[1] * self.A[-1].shape[1])
        self.W = np.random.normal(0, 1, size=sz)

        # init weight mask
        mask = list()
        for ia in range(len(self.A) - 1):
            cols = self.A[ia + 1].shape[1] - 1
            if ia == len(self.A) - 2:
                cols += 1
            for i in range(self.A[ia].shape[1]):
                if i == 0:
                    mask.extend(np.zeros(cols))
                else:
                    mask.extend(np.ones(cols))
        self.W_mask = np.array(mask, dtype=bool)
        return self.W

    def unflatten_coefs(self, coefs):
        """
        Unflatten vector of coefficient. Used for matrix propagation.
        :param coefs: vector of coefficients.
        :return: unflattened coefficients.
        """
        unflat_coefs = list()
        index = 0
        for ia in range(len(self.A) - 1):
            rows = self.A[ia].shape[1]
            cols = self.A[ia + 1].shape[1] - 1
            if ia == len(self.A) - 2:
                cols += 1
            res = list()
            for _ in range(rows):
                res.append(coefs[index:index + cols])
                index += cols
            unflat_coefs.append(np.array(res))
        return np.array(unflat_coefs)

    def fit(self, X, y):
        """
        Fit neural network on data X and target y.
        :param X: data matrix or vector.
        :param y:  target vector.
        :return: model parameters (coefficients).
        """
        self.set_data_(X, y)
        coefs = self.init_weights_  ()

        coefs = fmin_l_bfgs_b(self.cost,
                              x0=coefs,
                              fprime=self.grad)[0]
        self.coefs_ = coefs
        return coefs

    def predict(self, X):
        """
        Predict class for given rows in data X.
        :param X: matrix or vector of data.
        :return: class for every row.

        """
        self.X = X
        y = self.forward(self.coefs_)
        return np.argmax(y, axis=1)

    def predict_proba(self, X):
        """
        Predict probability for given data X.
        :param X: matrix or vector of data.
        :return: percentages for belonging to each class for every row.
        """
        self.X = X
        y = self.forward(self.coefs_)
        return normalize(y, axis=1, norm='l1')

    def grad_approx(self, coefs, e):
        """
        Calculate approximate gradient for given coefficients.
        :param coefs: vector of coefficients
        :param e: size of shift of coefficient
        :return: gradient vector
        """
        e_vec = np.zeros(len(coefs))
        grad = np.zeros(len(coefs))

        for i in range(len(coefs)):
            e_vec[i] = e
            loss_left = self.cost(coefs - e_vec)
            loss_right = self.cost(coefs + e_vec)

            grad[i] = (loss_right - loss_left) / (2 * e)
            e_vec[i] = 0

        return grad

    def grad(self, coefs):
        """
        Calculate gradient for coefficients based on derived formula.
        :param coefs: vector of coefficients
        :return: gradient for given coefficients
        """
        predicted = self.forward(coefs)
        W = self.unflatten_coefs(coefs)

        D = [None for _ in range(len(self.A) - 1)]
        m = self.X.shape[0]

        # backwards propagation
        for i in range(len(self.A) - 1, 0, -1):
            if i == len(self.A) - 1:  # last layer
                al = np.multiply(self.A[i], (1 - self.A[i]))
                dl = np.multiply((predicted - self.y_hot), al)
                w = W[i - 1]
                w[0] = np.zeros(w.shape[1])
                D[i - 1] = np.dot(self.A[i - 1].T, dl) / m + self.alpha * w
            else:
                al = np.multiply(self.A[i], (1 - self.A[i]))
                dl = np.multiply(np.dot(dl, W[i].T), al)[:, 1:]
                w = W[i - 1]
                w[0] = np.zeros(w.shape[1])
                D[i - 1] = np.dot(self.A[i - 1].T, dl) / m + self.alpha * w

        D = [d.flatten() for d in D]
        D = [item for sublist in D for item in sublist]
        return np.array(D)

    def forward(self, coefs):
        """
        Forward propagation of Neural network.
        :param coefs: vector of coefficients
        :return: result of last layer
        """
        W = self.unflatten_coefs(coefs)
        X = np.copy(self.X)
        ones = np.ones(X.shape[0])[np.newaxis].T

        X = np.hstack((ones, X))
        self.A[0] = X
        for i, w in enumerate(W):
            z = np.dot(X, w)
            X = self.sigmoid(z)
            if i != len(W) - 1:
                X = np.hstack((ones, X))
            self.A[i + 1] = X

        return X

    def cost(self, coefs):
        """
        Calculate cost function for given coefficients.
        :param coefs: vector of coefficients
        :return: cost
        """
        predicted = self.forward(coefs)
        m = self.X.shape[0]
        J = np.sum(np.sum((predicted - self.y_hot) ** 2)) / (2 * m)
        reg = (self.alpha / 2) * sum(coefs[self.W_mask] ** 2)
        return J + reg


def cross_validation(X, y, kfolds):
    """
    Cross validation of Neural network classification, logistic regression and gradient boosting classifier with F1 scoring.
    """
    from sklearn.model_selection import cross_validate, cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score

    nna = NeuralNetwork([20, 20, 20], alpha=1e-5)
    nna_pred = cross_val_predict(nna, X, y, cv=kfolds)
    nna_score = f1_score(y, nna_pred, average='macro')

    lr = LogisticRegression()
    lr_pred = cross_val_predict(lr, X, y, cv=kfolds)
    lr_score = f1_score(y, lr_pred, average='macro')

    grc = GradientBoostingClassifier()
    grc_pred = cross_val_predict(grc, X, y, cv=kfolds)
    grc_score = f1_score(y, grc_pred, average='macro')

    print('F1 scores: Neural network: {nna} | Logistic regression: {lr} | Gradient boosting classifier: {gbc}'.format(
        nna=nna_score, lr=lr_score, gbc=grc_score))


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    #data_full = datasets.load_digits()
    #X, y = data_full.data, data_full.target

    cross_validation(X, y, 5)
