class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param X: pd.DataFrame|list - train input/messages
        :param y: pd.DataFrame|list - train output/labels
        :return: None
        """
        pass

    def predict_prob(self, message, label):
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        pass

    def predict(self, message):
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
        """
        pass

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param X: pd.DataFrame|list - test data - messages
        :param y: pd.DataFrame|list - test labels
        :return:
        """
        pass
