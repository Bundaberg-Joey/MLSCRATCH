import numpy as np


class NaiveBayesClassifier(object):
    """Basic Implementation for continuous variables, assuming independant and normally distributed.
    Attributes
    ----------
    class_prob : dict, {class[int]: probability[float]}
        Keys are classes present in training data, values are the log probability of selecting that class at random.
    X_mu : dict, {class[int]: np.array([float])}
        Keys are class keys, values are the arithmatic means of features for that class.
    X_sigma : dict, {class[int]: np.array([float])}
        Keys are class keys, values are the standard deviations of features for that class.
    Methods
    -------
    Notes
    -----
    Summation of log probabilities is used for numerical stability than the standard cumulative product approach.
    While not strictly necessary in NaiveBayes, this implementation calculates the means and standard deviations of each
    feature for each class during fitting.
    This saves the algorithm having to continually calculate these values during prediction and means we still keep the
    `learning` aspect of machine learning.
    """

    def __init__(self, prob_func=None):
        """Initialise with pdf function
        Parameters
        ----------
        prob_func : func(x, mu, sigma) (default = False)
            Function which can take a singular point and calculate the pdf at that point.
            If not passed then inbuilt gaussian pdf is used,
        """
        self.prob_func = self._gauss_pdf if not prob_func else prob_func
        self.class_prob = {}
        self.X_mu = {}
        self.X_sigma = {}

    @staticmethod
    def _prob_class(y):
        """Calculates plog frequency of classes in given training set.
        Parameters
        ----------
        y : np.array(), shape (num entries,)
            Target classifications, must be integers for later indexing to work.
        Returns
        -------
        cf_dict : dict, {class[int]: probability[float]}
            Keys are classes present in training data, values are the log probability of selecting that class at random.
        """
        classes, counts = np.unique(y, return_counts=True)
        frequency = counts / len(y)
        cf_dict = {c: np.log(f) for c, f in zip(classes, frequency)}
        return cf_dict

    def _prob_feature_class(self, X, y):
        """Calculates mu and std of each feature for each class.
        Parameters
        ----------
        X : np.array(), shape (num_entries, num_features)
            Feature matrix containing numerical values only.
        y : np.array(), shape (num entries,)
            Target classifications, must be integers
        Returns
        -------
        mus, sigmas : (dict, dict), ({class[int]: np.array([float])}, {class[int]: np.array([float])})
            Keys are class keys, values are the means / standard deviations respectively of features for that class.
        """
        mus = {c: [] for c in self.class_prob}
        sigmas = dict(mus)

        for c in self.class_prob:
            X_c = X[y == c]
            mus[c] = X_c.mean(axis=0)  # mean of the columns
            sigmas[c] = X_c.std(axis=0)
        return mus, sigmas

    @staticmethod
    def _gauss_pdf(x, mu, sigma):
        """Calculates the gaussian PDF for the passed points.
        Parameters
        ----------
        x : float
            x coordinate to assess.
        mu : float
            mean of the gaussian distribution.
        sigma : float
            standard deviation of the gaussian distribution.
        Returns
        -------
        pdf : float
            probability of point `x` being in the distribution.
        """
        a = np.divide(1, (np.sqrt(2 * np.pi)) * sigma)
        b = np.exp(np.divide(-np.square(x - mu), 2 * np.square(sigma)))
        pdf = a * b
        return pdf

    def _predict_point(self, x):
        """Classifies a point given the probability of it belonging to a class.
        For each class the summation of the log probabilities are compared and the largest value is selected as label.
        The probabillity is calculated by determing the pdf of the features of `x` for each class.
        Parameters
        ----------
        x : np.array(), shape (num_features, )
            Featuve vector.
        Returns
        -------
        predicted class : int
            The iteger label of the predicted class.
        """
        cX = []
        for c in self.class_prob:
            a, b = self.class_prob[c], 0
            for pos, feature_val in enumerate(x):
                mu, std = self.X_mu[c][pos], self.X_sigma[c][pos]
                prob = self.prob_func(feature_val, mu, std)
                b += np.log(prob)
            cX.append(a + b)

        return np.argmax(cX)

    def fit(self, X_train, y_train):
        """Trains the classifier at initialisation.
        Parameters
        ----------
        X_train : np.array(), shape (num_entries, num_features)
            Feature matrix containing numerical values only.
        y_train : np.array(), shape (num entries,)
            Target classifications, must be integers for later indexing to work.
        """
        X_train, y_train = np.array(X_train), np.array(y_train).astype(int)
        self.class_prob = self._prob_class(y_train)
        self.X_mu, self.X_sigma = self._prob_feature_class(X_train, y_train)

    def predict(self, X):
        """Predicts the class of passed feature matrix `X`.

        Parameters
        ----------
        X : np.array(), shape(num_entries, num_features)
            Feature matrix.
        Returns
        -------
        predictions, np.array(), shape(num_entries, )
            Array containing integeres corresponding to the predicted labels.
        """
        X = np.array(X)
        n = len(X.shape)
        if n > 1:
            prediction = [self._predict_point(x) for x in X]
        elif n == 1:
            prediction = self._predict_point(X)
        else:
            prediction = []
        return np.array(prediction)
