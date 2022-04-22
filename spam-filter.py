'''
Coursework Problem 7.3

Implement a spam filter that is based on naive Bayes.
'''

import csv
import re


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
        print("Spam")
    elif p_spam_given_message < p_ham_given_message:
        print("Ham")
    else:
        print("Equal probs")

    print(f"Probability of spam: {p_spam_given_message}")
    print(f"Probability of ham: {p_ham_given_message}")


if __name__ == "__main__":
    header, df = import_dataset("data/youtube_combined.csv")
    cleaned_df = clean_dataset(df)
    vocab = generate_vocabulary(cleaned_df)
    spam, ham = seperate_spam_ham(cleaned_df)
    constants = calculate_constants(spam, ham, vocab)
    spam_parameters, ham_parameters = calculate_parameters(
        spam, ham, vocab, constants)

    classify("i love this video", constants, spam_parameters, ham_parameters)
