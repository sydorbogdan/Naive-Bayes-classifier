import numpy as np
import pandas as pd
import re


class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """
    def __init__(self):
        self.features = {}
        self.label_probs = {}
        self.feature_probs = []
        self.conditional_probs = {}

    def fit(self, X, y):
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param X: pd.DataFrame|list - train input/messages
        :param y: pd.DataFrame|list - train output/labels
        :return: None
        """
        self.features = {feature: ind for ind, feature
                         in enumerate(np.unique(np.hstack(X['Processed Tweet'])))}
        labels =  y['label'].unique()
        self.label_probs = {label: 0 for label in labels}
        self.conditional_probs = {label: [0] * len(self.features) for label in labels}

        print(self.features)
        print(len(self.features))
        def check_features(row):
            processed_tweet = row["Processed Tweet"]
            for word in set(processed_tweet):
                self.conditional_probs[row['label']][self.features[word]] += 1
            self.label_probs[row['label']] += 1
            return row

        df = X.merge(y, left_index=True, right_index=True).apply(check_features, axis=1)

        for label in self.label_probs:
            self.conditional_probs[label] = [prob / self.label_probs[label]
                                         for prob in self.conditional_probs[label]]
            self.label_probs[label] /= len(df)
        self.feature_probs = [0] * len(self.features)
        for count in range(len(self.features)):
            self.feature_probs[count] = sum([self.conditional_probs[label][count] * self.label_probs[label]
                                             for label in self.label_probs])


        print(self.features)
        print(self.feature_probs)
        print(self.label_probs)
        print(self.conditional_probs)



    def predict_prob(self, message, label):
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        tweet_words = {word for word in re.sub(r"[\W\d]", r" ", message).split()
                 if word in self.features}
        features = [0] * len(self.features)
        for word in tweet_words:
            features[self.features[word]] = 1
        condititional_prob = 1
        total_prob = 1
        for count in range(len(features)):
            if features[count] == 1:
                condititional_prob *= self.conditional_probs[label][count]
                total_prob *= self.feature_probs[count]
            else:
                condititional_prob *= 1 - self.conditional_probs[label][count]
                total_prob *= 1 - self.feature_probs[count]
        return (condititional_prob * self.label_probs[label]) / total_prob


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
