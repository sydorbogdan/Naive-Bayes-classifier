from bayesian_classifier import BayesianClassifier

import pandas as pd
import re


def process_data(data_file, stop_words_file="stop_words.txt"):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    with open(stop_words_file) as in_file:
        stop_words = in_file.read().split('\n')
    word_frequencies = {}

    def clear_text(row, min_len=4):
        text = re.sub(r"[\W\d]", r" ", row["tweet"])
        print(text)
        row["Processed Tweet"] = [word for word in text.split()
                                  if word not in stop_words and len(word) >= min_len]
        for word in row["Processed Tweet"]:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
        return row

    def remove_infrequent(row, min_freq=10):
        row["Processed Tweet"] = [word for word in row["Processed Tweet"]
                                  if word_frequencies[word] >= min_freq]
        return row

    df = pd\
        .read_csv(data_file, encoding="utf8")\
        .apply(clear_text, axis=1)\
        .drop(labels=["tweet", "id", "Unnamed: 0"], axis=1)\
        .apply(remove_infrequent, axis=1)


    return df.drop("label", axis=1), df.drop("Processed Tweet", axis=1)


if __name__ == "__main__":
    train_X, train_y = process_data("train.csv")
    #test_X, test_y = process_data("your test data file")

    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    #classifier.predict_prob(test_X[0], test_y[0])

    #print("model score: ", classifier.score(test_X, test_y))
    #print(process_data("train.csv"))