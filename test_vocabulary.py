import string
# !!!!!!!!!!!!!!!! Uncomment if running first time
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from bayesian_classifier import BayesianClassifier
import re
import time





if __name__ == "__main__":
    start = time.time()
    train_X, train_y = process_data("train.csv")
    print(time.time() - start)
    # test_X, test_y = process_data("your test data file")
    # 35.09407687187195

    # classifier = BayesianClassifier()
    # classifier.fit(train_X, train_y)
    # classifier.predict_prob(test_X[0], test_y[0])

    # print("model score: ", classifier.score(test_X, test_y))
    # print(process_data("train.csv"))