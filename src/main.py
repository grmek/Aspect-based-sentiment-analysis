import os
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from lexicon_features_model import LexiconFeaturesModel


def main():
    """
        MAIN OHOHOHOHOHO
    """

    data_dir = '../data/'
    data_set_dir = data_dir + 'SentiCoref_1.0'
    sentiment_lexicon_dir = data_dir + 'Lexicon'

    sentiment_lexicon_dict = get_sentiment_lexicon(sentiment_lexicon_dir)
    dataset, sentiments, file_names = create_dataset_and_sentiments(data_set_dir)

    print_statistics(dataset, sentiments, "Whole dataset")

    # train test split
    dataset_train, dataset_test, sentiments_train, sentiments_test = train_test_split(dataset, sentiments,
                                                                                      test_size=0.33, random_state=42)

    print_statistics(dataset_test, sentiments_test, "Test dataset")

    lexf_model = LexiconFeaturesModel(sentiment_lexicon_dict)
    lexf_model.fit(dataset_train, sentiments_train)
    lexf_predicted = lexf_model.predict(dataset_test, sentiments_test)

    random_predicted = predict_random(sentiments_test)
    neutral_predicted = predict_majority(sentiments_test, 3)

    print_statistics(dataset_test, lexf_predicted, "Lexicon Features Model predicted")
    print_statistics(dataset_test, random_predicted, "Random predicted")
    print_statistics(dataset_test, neutral_predicted, "Majority predicted")

    calculate_measures(sentiments_test, lexf_predicted, print_model_name='Lexicon Features Model')
    calculate_measures(sentiments_test, random_predicted, print_model_name='Random Model')
    calculate_measures(sentiments_test, neutral_predicted, print_model_name='Majority Model')


def get_sentiment_lexicon(lexicon_dir):
    """
        Read all lexicon files
    :param lexicon_dir: directory path of all lexicon files
    :return: sentiment_lexicon: dictionary (key: <word>, value: <sentiment>)
                        (here <sentiment> has values -1 if negative and 1 if positive sentiment)
    """

    sentiment_lexicon = dict()
    for file_name in os.listdir(lexicon_dir):
        curr_sentiment = 1  # indicates positive sentiment
        if file_name.split('_')[0] == 'negative':
            curr_sentiment = -1  # indicates negative sentiment
        with open(os.path.join(lexicon_dir, file_name), encoding='utf-8') as f:
            for word in f:
                sentiment_lexicon[word.strip()] = curr_sentiment

    return sentiment_lexicon


def get_data_sample(file_path):
    """
        Read all .tsv files
    :param file_path: path of .tsv file
    :return: data sample
    """
    with open(file_path, encoding='utf-8') as f:
        return [line[:-2].split('\t') for line in f.readlines()[7:]]


def create_dataset_and_sentiments(data_set_dir):
    """
        Read all .tsv files from directory
    :param      data_set_dir: directory path of all .tsv files
    :return:    data_set: dataset of samples described in transform_sample function
                sentiments: list of ground truth dictionaries
    """
    # total_deviation = 0
    # num_comparisons = 0
    data_set = []
    sentiments = []
    file_names = []

    for file_name in os.listdir(data_set_dir):
        file_names.append(file_name)
        data_sample = get_data_sample(data_set_dir + '/' + file_name)
        data_row, ground_truth = transform_sample(data_sample)
        data_set.append(data_row)
        sentiments.append(ground_truth)

    return data_set, sentiments, file_names


def transform_sample(data_sample):
    """
        Transform data sample into X sample row and y ground truth dictionary
    :param  data_sample: array from a line from .tsv file
    :return data: one array sample for X dataset  -> [ [ <word> , <array of entitiesID> ] ,... ]
                        (majority have empty array in <array of entities>)
            sentiment_dict: ground truth dictionary (key: <entityID>, value: <sentiment>)
                        (here <sentiment> has range from 1 to 5)
    """

    data = []
    sentiment_dict = dict()

    for idx, entry in enumerate(data_sample):
        word = entry[2]
        entities = []

        if entry[-1] != '_':
            for entity_str in entry[-1].split('|'):
                entities.append(int(entity_str[2:-1]))

        if entry[-3] != '_':
            sentiments = entry[-3].split('|')
            while len(sentiments) < len(entities):  # sometimes only one sentiment is defined where multiple entities
                sentiments.append(sentiments[-1])

            for entity, sentiment in zip(entities, sentiments):
                if entity != '_':
                    sentiment_dict[entity] = int(sentiment[0])
        data.append([word, entities])
    return data, sentiment_dict


def get_ground_truth(data_sample):
    sentiment_dict = dict()

    for entry in data_sample:
        if entry[-3] != '_':
            entities = entry[-1].split('|')
            sentiments = entry[-3].split('|')
            while len(sentiments) < len(entities):
                sentiments.append(sentiments[-1])
            for entity, sentiment in zip(entities, sentiments):
                if entity != '_':
                    sentiment_dict[int(entity[2:-1])] = int(sentiment[0])

    return sentiment_dict


def predict_random(sentiment_dicts):
    """
        Randomly predict sentiments
    :param      sentiment_dicts: list of dictionaries
    :return:    random predictions
    """
    return predict_dummy(sentiment_dicts)


def predict_majority(sentiment_dicts, pred):
    """
        Majority prediction
    :param sentiment_dicts: list of dictionaries
    :param pred: majority class
    :return: all predictions are equal pred
    """
    return predict_dummy(sentiment_dicts, pred)


def predict_dummy(sentiment_dicts, pred=None):
    result = []
    for sen_dict in sentiment_dicts:
        d = dict()
        for entity in sen_dict.keys():
            if pred is None:
                d[entity] = random.randint(1, 5)
            else:
                d[entity] = pred
        result.append(d)
    return result


def print_statistics(dataset, sentiment_dicts, first_text):
    """
        Print different statistics
    :param dataset: list of data
    :param sentiment_dicts: list of dictionaries
    :param first_text: to print model name
    """
    num_classes = np.zeros(5)
    for sentiment_dict in sentiment_dicts:
        sentiments = sentiment_dict.values()
        for s in sentiments:
            num_classes[s-1] += 1

    print(first_text, " statistics:")
    print("Number of sentiments from 1 to 5", num_classes)


def calculate_measures(sentiment_dicts_true, sentiment_dicts_pred, print_model_name=None):
    """
        Final measures to compare different models
    :param sentiment_dicts_true: true sentiments
    :param sentiment_dicts_pred: predicted sentiments
    :param print_model_name: to print model name
    :return: accuracy, precision, recall, f1 scores
    """
    num_samples = sum([len(dictionary) for dictionary in sentiment_dicts_true])
    y_true = np.zeros(num_samples)
    y_pred = np.zeros(num_samples)
    i = 0
    for dict_true, dict_pred in zip(sentiment_dicts_true, sentiment_dicts_pred):
        entities = list(dict_true.keys())
        for entity in entities:
            y_true[i] = dict_true[entity]
            y_pred[i] = dict_pred[entity]
            i += 1

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    if print_model_name is not None:
        print()
        print("Measures for {}:".format(print_model_name))
        print("Accuracy: ", acc)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print('F1 score: ', f1)
    return acc, precision, recall, f1


if __name__ == '__main__':
    main()
