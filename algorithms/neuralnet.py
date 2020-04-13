import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworkClassifier(object):

    def __init__(self, learning_rate, architecture, its_per_check, num_iterations):
        """
        Rudimentary implementation of a Neural Network Classifier in numpy.
        NN parameters are fitted by gradient descent / back propagation without regularisation.
        Only tested for binary classification but architecture can be specified by user to be multiclassification.
        """
        self.learning_rate = learning_rate
        self.architecture = architecture  # list of num nodes in each layer (includes input / output layers)
        self.L = len(self.architecture)  # num layers
        self.theta, self.b = self._initial_network_parameters(self.architecture, self.L)
        self.its_per_check = its_per_check
        self.num_iterations = num_iterations
        self.theta_opt, self.b_opt = None, None  # placeholders for optimal values later on

    def _initial_network_parameters(self, architecture, num_layers):
        """
        Determines the initial weights and biases of the Neural Network depending on the given architecture.
        Random values are assigned to the weights and biases.
        :param architecture: list[int], integers denoting how many nodes present in each layer (includes input / output)
        :param num_layers: int, number of layers in the NN.
        :return: weights: list[np.array(m, n)], list contains the theta weight matrices for moving between NN layers
        :return: biases: list[np.array(m, 1)], list contains the bias values for each layer of the NN
        """
        weights, biases = [], []
        for i in range(num_layers - 1):
            w = np.random.randn(self.architecture[i+1], self.architecture[i])
            b = np.random.randn(self.architecture[i+1],1)
            weights.append(w)
            biases.append(b)
        return weights, biases

    def _sigmoid(self, z):
        """
        Calculates the logistic sigmoid of the passed vector.
        :param z: np.array(m), input vector to "flatten"
        :return: g: np.array(m), flattened vecotr w.r.t the sigmoid function
        """
        g = np.divide(1, 1 + np.exp(-z))
        return g

    def _forward_propagation(self, X, n):
        """
        Calculate the activation values of the NN by forwards propagation of the given features `X` through the NN.
        The activations of the layer are the sigmoid of the matrix multiplication of the prior layer's weights and
        activations plus the addition of the prior layer's bias terms.
        :param X: np.array(m, n), feature matrix
        :param n: int, number of features in the passed feature matrix
        :return: actv: list[np.array], each list element is the activation values of the NN for the given layer
        """
        actv = [X] + [[]] * (self.L-1)  # first activation layer is always the input features

        for layer in range(1, self.L):
            a = self._sigmoid(np.matmul(self.theta[layer-1], actv[layer-1]) + np.repeat(self.b[layer-1], n, 1))
            actv[layer] = a
        return actv

    def _backward_propagation(self, X, y, n, actv):
        """
        Determine the ideal parameters of the NN through backwards propagation.
        The final layer `Deltas` can be determined from the model output (final activations) and true values.
        :param X: np.array(m, n), feature matrix
        :param y: np.array(m), true values
        :param n: int, number of features in the passed feature matrix
        :param actv: list[np.array], each list element is the activation values of the NN for the given layer
        :return: Deltas: list[np.array], each list element is the Delta values of the NN for the given layer
        """
        D_L = -y * (1 - actv[self.L-1]) + (1 - y) * actv[self.L-1]  # final layer Deltas
        Deltas = [[]] * (self.L-1) + [D_L]

        for i in range(2,self.L):
            layer_delta = np.matmul(self.theta[self.L-i].T, Deltas[self.L - i+1]) * actv[self.L-i] * (1 - actv[self.L-i])
            Deltas[self.L-i] = layer_delta
        return Deltas

    def predict(self, X):
        """
        Given new feature data, predict the labels using the trained NN by forwards propagation.
        Forwards propagation of the new features through the NN to gain predictions is performed and the output layer
        of the activations of the NN is returned.
        :param X: np.array(m, n), feature matrix
        :return: y_pred: np.array(n), vector of predicted labels
        """
        n = X.shape[1]
        nn_activations = self._forward_propagation(X, n)
        y_pred = nn_activations[-1].reshape(n)
        return y_pred

    def _loss(self, X, y):
        """
        Calculate the NN loss between predicted values and true values.
        :param: X, np.array(m, n), feature matrix
        :param: y, np.array(m), target array
        :return: calcd_loss: float, calculated loss of the NN
        """
        p = self.predict(X)
        calcd_loss = np.mean(-y * np.log(p) - (1-y) * np.log(1-p))
        return calcd_loss

    def _train(self, X, y):
        """
        Performs an iteration of gradient descent whereby the weights and bias attributes of the NN are updated.
        First forward propagation is conducted to gain initial values of the NN, afterwhich the Delta losses are calcd.
        Then, for each layer in the NN the weights and biases are iteratively updated.
        :param: X, np.array(m, n), feature matrix
        :param: y, np.array(m), target array
        """
        n = X.shape[1]
        A = self._forward_propagation(X, n)
        D = self._backward_propagation(X, y, n, A)

        for i in range(self.L-1):
            self.theta[i] -= self.learning_rate * np.matmul(D[i+1], A[i].T)
            self.b[i] -= self.learning_rate * np.sum(D[i+1],1).reshape(-1,1)

    def fit(self, X, y):
        """
        Fit the NN to the passed feature and target data `X` and `y` to determine weights and biases.
        For each of the inputted iterations, perform `n` purely training iterations before checking if loss has lowered.
        This saves having to make the comparison eah time and increases the number of training cycles performed by NN.
        If the loss of the NN is lower than previously at the time of checking, then placeholder values are updated,
        which are then set to the main model parameters after fitting is completed.
        :param: X, np.array(m, n), feature matrix
        :param: y, np.array(m), target arrays
        """
        best = self._loss(X, y)

        for i in range(self.num_iterations):
            for updates in range(self.its_per_check):
                self._train(X, y)

            training_loss = self._loss(X, y)
            if training_loss < best:  # check if NN loss has decreased
                self.theta_opt, self.b_opt = self.theta, self.b
                best = training_loss

        self.theta, self.b = self.theta_opt, self.b_opt  # set default to be optimum


# test
np.random.RandomState(1)

# create training data and split into Train / Test blocks
n_points = 2000
train_size = int(0.7 * n_points)
A = np.random.uniform(-1, 1, (2, n_points))
u = 2
b = (np.sin(u*np.pi*A[0,:])*(np.sin(u*np.pi*A[1,:]))<0)*1
A_train, A_test, b_train, b_test = A[:,:train_size], A[:,train_size:], b[:train_size], b[train_size:]

# Initialise the model
mdl = NeuralNetworkClassifier(learning_rate=1e-3, architecture=[2,10,10,1], its_per_check=10, num_iterations=1500)

# Train the model
mdl.fit(A_train, b_train)

# make prediction from the model
b_pred = mdl.predict(A_test)
binary_pred = b_pred>=0.5
full_acc = sum(binary_pred == b_test) / len(b_test)
plt.figure(figsize=(5,4))
plt.title(F" Full NN accuracy = {full_acc:.3f}")
plt.scatter(A_test[0,:], A_test[1,:], 10, b_pred)
plt.show()