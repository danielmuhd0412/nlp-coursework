import csv
import re
import random

LETTERS = set("abcdefghijklmnopqrstuvwxyz")


def import_dataset(file, delim=","):
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
    vowel_char_map = {ord('ä'): 'ae', ord('ü'): 'ue',
                      ord('ö'): 'oe', ord('ß'): 'ss'}

    for row in df:
        row[1] = ''.join(row[1:])

        if len(row) > 2:
            del row[2:]

        row[1] = re.sub("\W", " ", row[1])  # remove non-words
        row[1] = re.sub("\s+", " ", row[1])  # remove extra whitespace
        row[1] = re.sub("\d+", " ", row[1])  # remove digits
        # remove non-latin characters
        row[1] = re.sub(r'[^\x00-\x7f]', r'', row[1])
        row[1] = row[1].lower().strip()
        row[1] = row[1].translate(vowel_char_map)  # replace umlauts: ö -> oe

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


def seperate_en_de(df):
    en, de = list(), list()

    for i in range(len(df)):
        text_class = df[i][0]
        text = df[i][1]

        if text_class == 'en':
            en.append(text)
        else:
            de.append(text)

    return en, de


def generate_letter_freqs(texts):
    # key = letter
    # value = array of frequencies of letter in each row of the dataset
    letter_counts = {letter: 0 for letter in LETTERS}

    for text in texts:
        letters = text.replace(" ", "")
        for letter in letters:
            letter_counts[letter] += 1  # increment frequency if letter found

    return letter_counts


def calculate_constants(en, de):
    p_en = len(en)
    p_de = len(de)

    n_en = 0  # sum of all letters in english
    for row in en:
        letters = row.replace(" ", "")
        n_en += len(letters)

    n_de = 0  # sum of all letters in german
    for row in de:
        letters = row.replace(" ", "")
        n_de += len(letters)

    n_letters = len(LETTERS)

    constants = {"p_en": p_en,
                 "p_de": p_de,
                 "n_en": n_en,
                 "n_de": n_de,
                 "n_letters": n_letters}

    return constants


def calculate_parameters(en, de, constants, alpha=1):
    en_parameters = dict()
    en_letter_freqs = generate_letter_freqs(en)

    de_parameters = dict()
    de_letter_freqs = generate_letter_freqs(de)

    # calculate p_letter_given_[en|de] for each word
    for letter in LETTERS:
        n_letter_given_en = en_letter_freqs[letter]
        en_parameters[letter] = (
            n_letter_given_en + alpha) / (constants["n_en"] + alpha * constants["n_letters"])

        n_letter_given_de = de_letter_freqs[letter]
        de_parameters[letter] = (
            n_letter_given_de + alpha) / (constants["n_de"] + alpha * constants["n_letters"])

    return en_parameters, de_parameters


def classify(message, constants, en_parameters, de_parameters):

    result = None

    # check input validity
    if type(message) is not str:
        print("Invalid input!")
        return None

    # clean input
    message = re.sub("\W", " ", message)
    message = re.sub("\s+", " ", message)
    message = message.lower().strip()

    letters = set(message.replace(" ", ""))

    p_en_given_message = constants["p_en"]
    p_de_given_message = constants["p_de"]

    for letter in letters:
        p_en_given_message *= en_parameters[letter]
        p_de_given_message *= de_parameters[letter]

    if p_en_given_message > p_de_given_message:
        result = 'en'
        # print("English")
    elif p_en_given_message < p_de_given_message:
        result = 'de'
        # print("German")
    else:
        pass
        # print("Equal probs")

    # print(f"Probability of English: {p_en_given_message}")
    # print(f"Probability of German: {p_de_given_message}")
    return result


def predict_results(test_df, constants, en_parameters, de_parameters):

    predicted_results = []
    actual_results = []

    for row in test_df:
        message = row[1]
        actual_result = row[0]

        predicted_result = classify(
            message, constants, en_parameters, de_parameters)
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

        if actual_results[i] == 'en':
            total_positives += 1

        if actual_results[i] == 'de':
            total_negatives += 1

        if predicted_results[i] == 'en' and actual_results[i] == 'en':
            true_positives += 1

        if predicted_results[i] == 'de' and actual_results[i] == 'de':
            true_negatives += 1

        if predicted_results[i] == 'de' and actual_results[i] == 'en':
            false_negatives += 1

        if predicted_results[i] == 'en' and actual_results[i] == 'de':
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

    print(
        f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score:{f1_score}")

if __name__ == "__main__":
    header_de, df_de = import_dataset("data/deu_news_2020_10K-sentences.csv")
    header_en, df_en = import_dataset("data/eng_news_2020_10K-sentences.csv")

    df = df_de + df_en

    cleaned_df = clean_dataset(df)
    randomised_df = randomise_dataset(cleaned_df)
    train_df, test_df = split_dataset(randomised_df)

    en, de = seperate_en_de(train_df)

    constants = calculate_constants(en, de)

    en_parameters, de_parameters = calculate_parameters(
            en, de, constants)

    predicted_results, actual_results = predict_results(
            test_df, constants, en_parameters, de_parameters)
    calculate_metrics(predicted_results, actual_results)