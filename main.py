from bayesian_classifier import BayesianClassifier

import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def remove_punctuation(inp_text: str) -> str:
    """
    Removes punctuation symbols from input text
    (probably don't work on low versions of python)
    :param inp_text: text which will be processed
    :return: text with removed punctuation
    """
    inp_text = inp_text.translate(str.maketrans('', '', string.punctuation))
    return inp_text


def remove_useless_words(inp_text, path):
    """
    Removes useless words fom text
    :param inp_text: text which will be processed
    :param path: path to file with useless word (words separated by \n symbol)
    :return: processed text
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(inp_text)
    processed_sentence = [w for w in word_tokens if w not in stop_words]
    return processed_sentence



def process_data(data_file, stop_words_file="stop_words.txt"):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    with open(stop_words_file) as in_file:
        stop_words = in_file.read().split('\n')
    def clear_text(row):
        text = re.sub(r"[#\",!\.\?]", r" ", row["tweet"])
        print(text)
        row["Processed Tweet"] = [word for word in text.split()
                                  if word not in stop_words]
        return row

    df = pd\
        .read_csv(data_file, encoding="utf8")\
        .drop("Unnamed: 0", axis=1)\
        .apply(clear_text, axis=1)
    return df.drop("label", axis=1), df.drop("Processed Tweet", axis=1)


if __name__ == "__main__":
    #train_X, train_y = process_data("your train data file")
    #test_X, test_y = process_data("your test data file")

    #classifier = BayesianClassifier()
    #classifier.fit(train_X, train_y)
    #classifier.predict_prob(test_X[0], test_y[0])

    #print("model score: ", classifier.score(test_X, test_y))
    print(process_data("train.csv"))