'''
Coursework Problem 7.3

Implement a spam filter that is based on naive Bayes.
'''

import csv
import re
import random


def import_dataset(file, headers=True, delim=","):
    df = list()

    with open(file, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=delim)
        for row in reader:
            df.append(row)

    # seperate header
    header = df[0]
    del df[0]

    return header, df


def clean_dataset(df):
    for row in df:
        del row[:3]  # remove COMMENT_ID, AUTHOR, DATE columns
        row[0] = re.sub(r"https?\S+", r"link", row[0])  # replace hyperlinks
        row[0] = re.sub("\W", " ", row[0])  # remove non-words
        row[0] = re.sub("\s+", " ", row[0])  # remove extra whitespace
        row[0] = row[0].lower().strip()

    return df


def randomise_dataset(df, seed=42):
    random.seed(seed)
    random.shuffle(df)

    return df


def split_dataset(df, ratio=0.7):

    split = int(ratio * len(df))

    train = df[:split]
    test = df[(split + 1):len(df)]

    return train, test


def generate_vocabulary(df):
    vocab = list()

    for row in df:
        words = row[0].split(" ")
        for word in words:
            if word not in vocab:
                vocab.append(word)

    return vocab


def generate_word_counts(messages, vocab):
    # key = word
    # value = array of frequencies of word in each row of the dataset
    word_counts = dict()

    for word in vocab:
        word_counts[word] = [0] * len(messages)

    for index, message in enumerate(messages):
        words = message.split(" ")
        for word in words:
            word_counts[word][index] += 1  # increment frequency if word found

    return word_counts


def seperate_spam_ham(df):
    spam, ham = list(), list()

    for i in range(len(df)):
        text = df[i][0]
        text_class = df[i][1]

        if text_class == '0':  # ham
            ham.append(text)
        else:
            spam.append(text)

    return spam, ham


def calculate_constants(spam, ham, vocab):
    p_spam = len(spam)
    p_ham = len(ham)

    n_spam = 0  # sum of all words in spam messages
    for row in spam:
        words = row.split(" ")
        n_spam += len(words)

    n_ham = 0  # sum of all words in ham messages
    for row in ham:
        words = row.split(" ")
        n_ham += len(words)

    n_vocab = len(vocab)

    constants = {"p_spam": p_spam,
                 "p_ham": p_ham,
                 "n_spam": n_spam,
                 "n_ham": n_ham,
                 "n_vocab": n_vocab}

    return constants


def calculate_parameters(spam, ham, vocab, constants, alpha=1):
    spam_parameters = dict()
    spam_word_counts = generate_word_counts(spam, vocab)

    ham_parameters = dict()
    ham_word_counts = generate_word_counts(ham, vocab)

    # calculate p_word_given_[spam|ham] for each word
    for unique_word in vocab:
        n_word_given_spam = sum(spam_word_counts[unique_word])
        spam_parameters[unique_word] = (
            n_word_given_spam + alpha) / (constants["n_spam"] + alpha * constants["n_vocab"])

        n_word_given_ham = sum(ham_word_counts[unique_word])
        ham_parameters[unique_word] = (
            n_word_given_ham + alpha) / (constants["n_ham"] + alpha * constants["n_vocab"])

    return spam_parameters, ham_parameters


def classify(message, constants, spam_parameters, ham_parameters):

    result = None

    # check input validity
    if type(message) is not str:
        print("Invalid input!")
        return None

    # clean input
    message = re.sub("\W", " ", message)
    message = re.sub("\s+", " ", message)
    message = message.lower().strip()

    words = message.split(" ")

    p_spam_given_message = constants["p_spam"]
    p_ham_given_message = constants["p_ham"]

    for word in words:
        if word in spam_parameters:
            p_spam_given_message *= spam_parameters[word]

        if word in ham_parameters:
            p_ham_given_message *= ham_parameters[word]

    if p_spam_given_message > p_ham_given_message:
        result = 1
        # print("Spam")
    elif p_spam_given_message < p_ham_given_message:
        result = 0
        # print("Ham")
    else:
        pass
        # print("Equal probs")

    # print(f"Probability of spam: {p_spam_given_message}")
    # print(f"Probability of ham: {p_ham_given_message}")
    return result


def predict_results(df, constants, spam_parameters, ham_parameters):

    predicted_results = []
    actual_results = []

    for row in test_df:
        message = row[0]
        actual_result = int(row[1])

        predicted_result = classify(
            message, constants, spam_parameters, ham_parameters)
        predicted_results.append(predicted_result)

        actual_results.append(actual_result)

    return predicted_results, actual_results


def calculate_metrics(predicted_results, actual_results):

    correct = 0
    total = len(predicted_results)

    true_positives = 0
    true_negatives = 0

    total_positives = 0
    total_negatives = 0

    false_positives = 0
    false_negatives = 0

    for i in range(len(predicted_results)):
        if predicted_results[i] == actual_results[i]:
            correct += 1

        if actual_results[i] == 1:
            total_positives += 1

        if actual_results[i] == 0:
            total_negatives += 1

        if predicted_results[i] == 1 and actual_results[i] == 1:
            true_positives += 1

        if predicted_results[i] == 0 and actual_results[i] == 0:
            true_negatives += 1

        if predicted_results[i] == 0 and actual_results[i] == 1:
            false_negatives += 1

        if predicted_results[i] == 1 and actual_results[i] == 0:
            false_positives += 1

    accuracy = correct / total
    precision = true_positives / total_positives
    recall = true_positives / (true_positives + false_negatives)
    f1_score = (2 * precision * recall) / (precision + recall)

    print(f"Total: {total}")
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Total Positives: {total_positives}")
    print(f"Total Negatives: {total_negatives}")

    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score:{f1_score}")


if __name__ == "__main__":
    header, df = import_dataset("data/youtube_combined.csv")
    cleaned_df = clean_dataset(df)
    randomised_df = randomise_dataset(cleaned_df)
    train_df, test_df = split_dataset(randomised_df)

    vocab = generate_vocabulary(train_df)
    spam, ham = seperate_spam_ham(train_df)
    constants = calculate_constants(spam, ham, vocab)
    spam_parameters, ham_parameters = calculate_parameters(
        spam, ham, vocab, constants)

    predicted_results, actual_results = predict_results(
        test_df, constants, spam_parameters, ham_parameters)
    calculate_metrics(predicted_results, actual_results)
