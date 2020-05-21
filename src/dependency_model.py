from general_model import GeneralModel
from collections import defaultdict
import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import stanza


def list_to_str(my_list, space=False):
    result = '['
    first = True
    for element in my_list:
        if first:
            first = False
        else:
            if space:
                result += ' '
            result += ','
        result += str(element)
    result += ']'
    return result


class DependencyModel(GeneralModel):
    def __init__(self, lexicon, positive_around_num=None, negative_around_num=None, normalize_data=False, rf_params=None):

        if positive_around_num is None:
            positive_around_num = [5]  # default positive environment ranges
        if negative_around_num is None:
            negative_around_num = [5]  # default negative environment ranges
        self.positive_around_num = positive_around_num
        self.negative_around_num = negative_around_num
        if rf_params is None:
            rf_params = {}          # default RF params -> then we use Grid Search

        self.feature_names = []
        for pos in positive_around_num:
            self.feature_names.append(f'+-{pos} positive')
        for neg in negative_around_num:
            self.feature_names.append(f'+-{neg} negative')

        self.feature_names = self.feature_names + [
            'closest positive',
            'closest negative',
            'positive in sentence',
            'negative in sentence',
            'all positive',
            'all negative',
            'all entities',
            'all entity occurrences',
            'positive - negative in sentence'
        ]
        self.num_features = len(self.feature_names)

        param_str = '_' + list_to_str(self.positive_around_num) + \
                    '_' + list_to_str(self.negative_around_num)
        if normalize_data:
            param_str += '_T'
        else:
            param_str += '_F'
        self.name = self.__class__.__name__ + '_' + param_str

        self.lexicon = lexicon

        # initialize dependency parser
        stanza.download("sl")
        self.dep_parser = stanza.Pipeline("sl", tokenize_pretokenized=True)

        self.normalizer = None
        if normalize_data:
            self.normalizer = Normalizer()
        self.rf_gs = None
        self.model = None
        # initialize a model
        if len(rf_params) > 0:
            # create a new random forest classifier
            rf = RandomForestClassifier()
            # create a dictionary of all values we want to test for n_estimators
            nf = self.num_features
            params_rf = {'n_estimators': [5, 25, 50, 75, 100, 200],
                         'criterion': ['gini', 'entropy'],
                         'max_depth': [3, int(nf / 4), int(nf / 2), int(nf * 3 / 4), nf],
                         }
            # use grid search to test all values for n_estimators
            self.rf_gs = GridSearchCV(rf, params_rf, cv=5, iid=False)
        else:
            self.model = RandomForestClassifier(**rf_params)

    def get_name(self):
        return self.name

    def fit(self, dataset, sentiment_dicts):
        """
            Fit model
        :param dataset: list of data
        :param sentiment_dicts: list of ground truth dictionaries
        """
        self.print('Fitting started, fist create features')
        X_train, y_train, back_to_dicts = self.get_features(dataset, sentiment_dicts, is_train=True)

        if self.normalizer is not None:
            self.print('Normalizing')
            self.normalizer.fit(X_train)

        assert (self.rf_gs is None) != (self.model is None)
        if self.model is None:
            self.print('Searching for best model parameters')

            # fit model to training data
            self.rf_gs.fit(X_train, y_train)

            # save best model
            self.model = self.rf_gs.best_estimator_
            self.print('Best parametres:', self.rf_gs.best_params_)
        else:
            self.print('Lets fit RF model')
            self.model.fit(X_train, y_train)

        self.print('Feature importances:')
        feature_importances = sorted(zip(self.feature_names, self.model.feature_importances_), key=lambda x: -x[1])
        for feature, value in feature_importances:
            self.print(f'    {round(value, 2)} {feature}')

    def predict(self, dataset, sentiment_dicts):
        """
            Predict values from dataset
        :param dataset: list of data
        :param sentiment_dicts: list of entity dictionaries
        :return: predicted sentiments (list of dictionaries)
        """
        self.print('Predicting results ...')
        X_test, _, back_to_dicts = self.get_features(dataset, sentiment_dicts, is_train=False)
        # self.print('Features are created')

        if self.normalizer is not None:
            self.normalizer.fit(X_test)

        y_predicted = self.model.predict(X_test)
        return predicted_to_dict(y_predicted, back_to_dicts)

    def get_features(self, dataset, y_dict, is_train=False):
        """
            Transform original dataset into features that are used for our model.
            Initial idea is that each entity is described in one row.
            We extract features from entity index list of text and sentiment list.
            All features have name in FEATURE_NAMES list.
        :param dataset: list of data
        :param y_dict: list of entity dictionaries
        :param is_train: boolean
        :return: X, y, back_to dicts (array that helps transform y back to list of dictionaries)
        """
        text_entities = [[entry[1] for entry in text] for text in dataset]
        sentiment_dataset = create_sentiment_array(dataset, self.lexicon)
        num_samples = sum([len(dictionary) for dictionary in y_dict])

        sentence_detector = nltk.data.load("tokenizers/punkt/slovene.pickle")

        X = np.zeros((num_samples, self.num_features))
        y = None
        if is_train:
            y = np.zeros(num_samples)
        back_to_dicts = np.empty((num_samples, 2), dtype=int)

        start_index = 0
        for idx, (entry, dictionary) in enumerate(zip(dataset, y_dict)):
            entities = list(dictionary.keys())
            end_index = start_index + len(entities)

            entity_to_idx = dict()
            for i, ent in enumerate(entities):
                entity_to_idx[ent] = start_index + i

            back_to_dicts[start_index:end_index, 0] = idx
            back_to_dicts[start_index:end_index, 1] = entities

            if is_train:
                y[start_index:end_index] = [dictionary[entity] for entity in entities]

            sentence_index_array, sentence_starts, sentence_ends = detect_sentences(entry, sentence_detector)
            sentence_count = sentence_index_array[-1] + 1
            sentence_positive_sentiment_counts = [0] * sentence_count
            sentence_negative_sentiment_counts = [0] * sentence_count
            for i, sentiment in sentiment_dataset[idx]:
                if sentiment > 0:
                    sentence_positive_sentiment_counts[int(sentence_index_array[i])] += 1
                elif sentiment < 0:
                    sentence_negative_sentiment_counts[int(sentence_index_array[i])] += 1

            sentence_entities = defaultdict(list)
            sentence_entity_indexes = defaultdict(list)
            for i, (token, entity_list) in enumerate(entry):
                if len(entity_list) > 0:
                    sentence_entities[int(sentence_index_array[i])] += [entity_to_idx[ent]
                                                                        for ent in entity_list
                                                                        if ent in entities]
                    sentence_entity_indexes[int(sentence_index_array[i])].append(i)
            sentence_unique_entity_counts = defaultdict(int, {k : len(set(v)) for k, v in sentence_entities.items()})

            tokenized_sentences = [[x[0] for x in entry[sentence_starts[i]:sentence_ends[i]+1]] for i in range(sentence_count)]
            sentence_dependencies = self.parse_dependencies(tokenized_sentences)

            column_id = 0

            for i, sentiment in sentiment_dataset[idx]:
                i_sentence = sentence_index_array[i]
                closest_distance = 1000000000
                closest_entities = []
                col_id = column_id
                for i_ent in sentence_entity_indexes[i_sentence]:
                    entity_list = [entity_to_idx[ent] for ent in entry[i_ent][1] if ent in entities]
                    distance = dependency_tree_distance(sentence_dependencies[i_sentence],
                                                        i - sentence_starts[i_sentence],
                                                        i_ent - sentence_starts[i_sentence])
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_entities = entity_list
                    elif distance == closest_distance:
                        closest_entities += entity_list

                    # positive around
                    col_id = column_id
                    for pos_around_num in self.positive_around_num:
                        if sentiment > 0 and distance < pos_around_num:
                            X[entity_list, col_id] += 1
                        col_id += 1
                    # negative around
                    for neg_around_num in self.negative_around_num:
                        if sentiment < 0 and distance < neg_around_num:
                            X[entity_list, col_id] += 1
                        col_id += 1

                # closest positive
                if sentiment > 0:
                    X[closest_entities, col_id] += 1
                col_id += 1
                # closest negative
                if sentiment < 0:
                    X[closest_entities, col_id] += 1

            column_id += len(self.positive_around_num) + len(self.negative_around_num) + 2

            for i, pos_sent_count in enumerate(sentence_positive_sentiment_counts):
                if sentence_unique_entity_counts[i] > 0:
                    X[sentence_entities[i], column_id] = pos_sent_count / sentence_unique_entity_counts[i]
            pos_sentence_id = column_id
            column_id += 1
            for i, neg_sent_count in enumerate(sentence_negative_sentiment_counts):
                if sentence_unique_entity_counts[i] > 0:
                    X[sentence_entities[i], column_id] = neg_sent_count / sentence_unique_entity_counts[i]
            neg_sentence_id = column_id
            column_id += 1

            all_positive = sum([1 for s in sentiment_dataset[idx] if s[1] == 1])
            X[start_index:end_index, column_id] = all_positive
            column_id += 1
            all_negative = sum([1 for s in sentiment_dataset[idx] if s[1] == -1])
            X[start_index:end_index, column_id] = all_negative
            column_id += 1
            all_entities = len(entities)
            X[start_index:end_index, column_id] = all_entities
            column_id += 1
            X = get_entity_occurrences(X, text_entities[idx], entity_to_idx)
            column_id += 1
            # positive - negative in sentence, divided by entity count
            X[start_index:end_index, column_id] = X[start_index:end_index, pos_sentence_id] - X[start_index:end_index, neg_sentence_id]

            start_index = end_index

        return pd.DataFrame(X, columns=self.feature_names), y, back_to_dicts

    def parse_dependencies(self, tokenized_sentences):
        return [sentence.words for sentence in self.dep_parser(tokenized_sentences).sentences]


def dependency_tree_distance(words, i1, i2):
    if i1 == i2:
        return 0
    w1_ancestors = {}
    dist = 1
    i = words[i1].head - 1
    while i != -1:
        w1_ancestors[i] = dist
        dist += 1
        i = words[i].head - 1
    root_dist = dist - 1
    dist = 1
    i = words[i2].head - 1
    while i != -1:
        if i in w1_ancestors:
            return dist + w1_ancestors[i]
        dist += 1
        i = words[i].head - 1
    return dist + root_dist - 1


def predicted_to_dict(predicted, back_to_dicts):
    """
        Transform y back into list of dictionaries
    :param predicted: y
    :param back_to_dicts: array
    :return: list of dictionaries
    """
    result = []
    curr_idx = -1

    for (idx, entity), prediction in zip(back_to_dicts, predicted):
        if curr_idx < idx:
            curr_idx = idx
            result.append(dict())
        result[curr_idx][entity] = int(prediction)
    return result


def get_entity_occurrences(X, text_entities, entity_to_idx, column_in_x=9):
    """
        Count entity number of occurrences in a text
    :param X: write result here
    :param text_entities: array of entities in text
    :param entity_to_idx: map from entity index to index in X row
    :param column_in_x: X column
    :return:
    """
    previous = []
    for entities in text_entities:
        current = []
        for entity in entities:
            current.append(entity)
            if entity not in previous and entity in entity_to_idx:
                X[entity_to_idx[entity], column_in_x] += 1
        previous = current
    return X


def exists_entity_in_dict(entities, entity_to_idx):
    for entity in entities:
        if entity in entity_to_idx:
            return True
    return False


def create_sentiment_array(data_set, lexicon):
    """
        For each text find words that are in lexicon and use lexicon sentiment to create array of sentiments
    :param data_set: list of data
    :param lexicon: sentiment dictionary
    :return: sentiment_dataset: list of text sentiments [ [ (<wordIDX>, <sentiment>) ,... ] ,... ]
    """
    sentiment_dataset = []
    for text in data_set:
        sentiment_arr = []
        for idx, entry in enumerate(text):
            word = entry[0]
            sentiment = lexicon.setdefault(word, 0)
            if sentiment != 0:
                sentiment_arr.append((idx, sentiment))
        sentiment_dataset.append(sentiment_arr)
    return sentiment_dataset


def detect_sentences(entry, tokenizer):
    """
        Detect which tokens in a dataset entry belong to the same sentences.
    :param entry: single dataset entry
    :param tokenizer: nltk Punkt sentence tokenizer
    :return: numpy array containing the index of the sentence each token belongs to
    """
    tokens = [x[0] for x in entry]
    sentences = tokenizer.sentences_from_tokens(tokens)
    sentence_index_array = np.zeros(len(tokens), dtype=int)
    sentence_starts = []
    sentence_ends = []
    idx = 0
    for i, sentence in enumerate(sentences):
        s_start = idx
        s_end = idx + len(sentence)
        sentence_index_array[s_start:s_end] = i
        sentence_starts.append(s_start)
        sentence_ends.append(s_end)
        idx += len(sentence)
    return sentence_index_array, sentence_starts, sentence_ends
