from bayesian_classifier import BayesianClassifier

import pandas as pd


def process_data(data_file):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    


if __name__ == "__main__":
    train_X, train_y = process_data("your train data file")
    test_X, test_y = process_data("your test data file")

    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    classifier.predict_prob(test_X[0], test_y[0])

    print("model score: ", classifier.score(test_X, test_y))
