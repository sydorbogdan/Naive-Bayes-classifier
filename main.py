from bayesian_classifier import BayesianClassifier
import re
import pandas as pd

with open('stop_words.txt') as in_file:
    STOP_WORDS = tuple(in_file.read().split('\n'))


def process_train_data(data_file, stop_words=STOP_WORDS):
    """
    Function for training data processing and splitting it into X and y sets.
    :param stop_words_file:
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """

    word_discrim = set()
    def clear_text(row, min_len=4):
        """
        Clears text from unnecessary
        information and words of length 3 or less
        """
        text = re.sub(r"[^a-zA-Z]", r" ", row["tweet"])
        text = re.sub(r"(.)\1(\1+)", r"\1", text)
        row["Processed Tweet"] = [word for word in text.lower().split()
                                  if word not in stop_words and word != 'user' and
                                  len(word) >= min_len]
        if row['label'] == 'discrim':
            for word in row["Processed Tweet"]:
                word_discrim.add(word)
        return row

    def remove_neutral_words(row):
        """
        Removes words that don't appear in
        discriminating tweets (for better accuracy)
        """
        row["Processed Tweet"] = [word for word in row["Processed Tweet"]
                                  if word in word_discrim]
        return row

    df = pd \
        .read_csv(data_file, encoding="utf8") \
        .apply(clear_text, axis=1) \
        .drop(labels=["tweet", "id", "Unnamed: 0"], axis=1) \
        .apply(remove_neutral_words, axis=1)

    return df.drop("label", axis=1), df.drop("Processed Tweet", axis=1)



def test_BayesianClassifier():
    """
    Trains Bayesian Classifier on test data and then tests it
    """
    train_X, train_y = process_train_data("train.csv")
    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    test_data = pd \
                .read_csv("test.csv", encoding="utf8") \
                .drop(labels=["id", "Unnamed: 0"], axis=1)
    test_X = test_data.drop("label", axis=1)
    test_y = test_data.drop("tweet", axis=1)

    print("model score: ", classifier.score(test_X, test_y) * 100, "%")



if __name__ == "__main__":
    test_BayesianClassifier()
