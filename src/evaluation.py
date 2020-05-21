import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
                            precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning
import warnings


class Evaluation:
    def __init__(self, dataset_dir):
        dataset, sentiments, file_names = create_dataset_and_sentiments(dataset_dir)
        print_statistics(dataset, sentiments, "Whole dataset")

        # train test split
        self.dataset_train,\
        self.dataset_test,\
        self.sentiments_train,\
        self.sentiments_test = train_test_split(dataset, sentiments, test_size=0.33, random_state=42)
        print_statistics(self.dataset_test, self.sentiments_test, "Test dataset")
        print()

        # don't show the UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    def evaluate(self, model):
        model.fit(self.dataset_train, self.sentiments_train)
        predicted = model.predict(self.dataset_test, self.sentiments_test)
        print_statistics(self.dataset_test, predicted, model.get_name() + " predicted")
        calculate_measures(self.sentiments_test, predicted, print_model_name=model.get_name())
        print()


def get_data_sample(file_path):
    """
        Read all .tsv files
    :param file_path: path of .tsv file
    :return: data sample
    """
    with open(file_path, encoding='utf-8') as f:
        return [line[:-2].split('\t') for line in f.readlines()[7:]]


def create_dataset_and_sentiments(dataset_dir):
    """
        Read all .tsv files from directory
    :param      dataset_dir: directory path of all .tsv files
    :return:    dataset: dataset of samples described in transform_sample function
                sentiments: list of ground truth dictionaries
    """
    # total_deviation = 0
    # num_comparisons = 0
    dataset = []
    sentiments = []
    file_names = []

    for file_name in os.listdir(dataset_dir):
        file_names.append(file_name)
        data_sample = get_data_sample(dataset_dir + '/' + file_name)
        data_row, ground_truth = transform_sample(data_sample)
        dataset.append(data_row)
        sentiments.append(ground_truth)

    return dataset, sentiments, file_names


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

    print(first_text, "statistics:")
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

    confusion_mtx = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    if print_model_name is not None:
        print('Measures for {}:'.format(print_model_name))
        print('Confusion matrix:')
        print(confusion_mtx)
        print('Accuracy:')
        print(acc)
        print('F1 score:')
        print(f1)
        print('Precision:')
        print(precision)
        print('Recall:')
        print(recall)
        print('RMSE:')
        print(rmse)

    return confusion_mtx, acc, f1, precision, recall, rmse
