import numpy as np
import pandas as pd
import re


class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """

    def __init__(self) -> None:
        """
        Initializes BayesianClassifier with label, features set to None
        and total score set to 0
        """
        self.labels = None
        self.features = None
        self.total_score = 0

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit Naive Bayes parameters according to train data X and y.
        """
        self.total_score = 0
        words = np.unique(np.hstack(X['Processed Tweet']))
        df_data = {'Feature': words, 'Index': range(len(words))}
        self.labels = {label: len(y[y['label'] == label]) / len(y)
                       for label in y['label'].unique()}

        df_data.update({label: [0] * len(words) for label in self.labels})
        self.features = pd.DataFrame(data=df_data).set_index("Feature")
        self.data_size = len(X)

        def check_features(row):
            """
            Calculates occurences of all feature words
            in the column of tweets
            """
            processed_tweet = row["Processed Tweet"]
            for word in processed_tweet:
                self.features.loc[word][row['label']] += 1
            return row

        df = X\
             .merge(y, left_index=True, right_index=True)\
             .apply(check_features, axis=1)
        for label in self.labels:
            self.features[label] = self.features[label].astype(float)

        def apply_log(row, big_num=-100000000):
            """
            Converts all numbers of occurrences
            to their log2
            (if number is 0 converts it to a
            big negative number)
            """
            for label in self.labels:
                if row[label] == 0.0:
                    row[label] = big_num
                else:
                    row[label] = np.log2(row[label])
            return row

        self.features = self.features.apply(apply_log, axis=1)

        for label in self.labels:
            self.features[label] -= np.log2(len(y[y['label'] == label]))

    def predict_prob(self, message: str, label: str) -> float:
        """
        Calculate the log2 of probability
        that a given label can be assigned to a given message.
        """
        message = re.sub(r"[^a-zA-Z\']", r" ", message)
        message = re.sub(r"(.)\1(\1+)", r"\1", message)
        tweet_words = {word for word in message.split()
                       if word in self.features.index.values.tolist()}
        conditional_prob = 0
        for word in tweet_words:
            conditional_prob += (self.features.loc[word][label]) + np.log2(self.labels[label])
        return conditional_prob - np.log2(self.data_size)


    def predict(self, message: str) -> str:
        """
        Predict label for a given message.
        """
        print(message)
        label_probs = [(label, self.predict_prob(message, label)) for label in self.labels]
        print(label_probs)
        max_value = max(label_probs, key=lambda x: x[1])[1]
        max_labels = [item[0] for item in label_probs if item[1] == max_value]

        return max(max_labels, key=lambda x: self.labels[x])


    def score(self, X: pd.DataFrame, y: pd. DataFrame) -> float:
        """
        Return the mean accuracy on the given test data
        and labels - the efficiency of a trained model.
        """
        self.total_score = 0
        def test_prediction(row):
            """
            Tests prediction for the given row
            """
            if self.predict(row['tweet']) == row['label']:
                self.total_score += 1

        df = X.merge(y, left_index=True, right_index=True).apply(test_prediction, axis=1)
        return self.total_score/len(df)

