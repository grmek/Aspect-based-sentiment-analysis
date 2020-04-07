import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

FEATURE_NAMES = [
    '+-5 positive',
    '+-5 negative',
    'closest positive',
    'closest negative',
    'positive in sentence',
    'negative in sentence',
    'all positive',
    'all negative',
    'all entities',
    'all entity occurrences'
]


class LexiconFeaturesModel:
    def __init__(self, lexicon):
        self.num_features = len(FEATURE_NAMES)
        self.lexicon = lexicon

        # initialize a model
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
        self.model = None

    def fit(self, dataset, sentiment_dicts):
        """
            Fit model
        :param dataset: list of data
        :param sentiment_dicts: list of ground truth dictionaries
        """
        print('[LexiconFeatures] Fitting started')
        X_train, y_train, back_to_dicts = self.get_features(dataset, sentiment_dicts, is_train=True)
        print('[LexiconFeatures] Features are created, now searching for best model parameters')

        # fit model to training data
        self.rf_gs.fit(X_train, y_train)

        # save best model
        self.model = self.rf_gs.best_estimator_
        print('[LexiconFeatures] Best parametres:', self.rf_gs.best_params_)

    def predict(self, dataset, sentiment_dicts):
        """
            Predict values from dataset
        :param dataset: list of data
        :param sentiment_dicts: list of entity dictionaries
        :return:
        """
        print('[LexiconFeatures] Predicting')
        X_test, _, back_to_dicts = self.get_features(dataset, sentiment_dicts, is_train=False)
        print('[LexiconFeatures] Features are created')

        y_predicted = self.model.predict(X_test)
        print('[LexiconFeatures] Finished xD')
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

            X = get_sentiment_around(X, text_entities[idx], sentiment_dataset[idx], entity_to_idx)
            X = get_closest_entities(X, text_entities[idx], sentiment_dataset[idx], entity_to_idx)
            X = get_single_entity_in_sentence(X, dataset[idx], sentiment_dataset[idx], entity_to_idx)
            all_positive = sum([1 for s in sentiment_dataset[idx] if s[1] == 1])
            X[start_index:end_index, 6] = all_positive
            all_negative = sum([1 for s in sentiment_dataset[idx] if s[1] == -1])
            X[start_index:end_index, 7] = all_negative
            all_entities = len(entities)
            X[start_index:end_index, 8] = all_entities
            X = get_entity_occurrences(X, text_entities[idx], entity_to_idx)

            start_index = end_index

        return pd.DataFrame(X, columns=FEATURE_NAMES), y, back_to_dicts


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


def get_sentiment_around(X, text_ent, sentiment_arr, entity_to_idx, pos_column_in_x=0, neg_column_in_x=1, around_num=5):
    """
        Count all sentiments that are +- around_num far from entity in a text
    :param X: write result here
    :param text_ent: array of entities in text
    :param sentiment_arr: sentiment array of text
    :param entity_to_idx: map from entity index to index in X row
    :param pos_column_in_x: X column that indicates positive sentiment
    :param neg_column_in_x: X column that indicates negative sentiment
    :param around_num: max distance from sentiment to entity in text
    :return: X
    """
    for idx, sentiment in sentiment_arr:
        column_in_x = pos_column_in_x
        if sentiment == -1:
            column_in_x = neg_column_in_x

        already_counted_entities = []
        for around in range(max(0, idx - around_num), min(idx + around_num + 1, len(text_ent))):
            if len(text_ent[around]) > 0:
                for entity in text_ent[around]:
                    if entity not in already_counted_entities:
                        already_counted_entities.append(entity)
                        if entity not in entity_to_idx:
                            # print("Entiteta {} ni v entity to idx listi?!?!?".format(entity))
                            pass
                        else:
                            X[entity_to_idx[entity], column_in_x] += 1

    return X


def get_closest_entities(X, text_ent, sentiment_arr, entity_to_idx, pos_column_in_x=2, neg_column_in_x=3):
    """
        For each sentiment find the closest entity/ies and add 1 to that entity/ies in appropriate cell of X
    :param X: write result here
    :param text_ent: array of entities in text
    :param sentiment_arr: sentiment array of text
    :param entity_to_idx: map from entity index to index in X row
    :param pos_column_in_x: X column that indicates positive sentiment
    :param neg_column_in_x: X column that indicates negative sentiment
    :return: X
    """
    for idx, sentiment in sentiment_arr:
        column_in_x = pos_column_in_x
        if sentiment == -1:
            column_in_x = neg_column_in_x

        closest_distance = 1000000000
        # closest_idx = 0
        closest_entities = []
        i = idx
        while i >= 0:
            i -= 1
            if len(text_ent[i]) > 0 and exists_entity_in_dict(text_ent[i], entity_to_idx):
                closest_distance = idx - i
                # closest_idx = i
                for ent in text_ent[i]:
                    closest_entities.append(ent)
                break

        i = idx + 1
        while i < min(len(text_ent), idx + closest_distance + 1):
            if len(text_ent[i]) > 0 and exists_entity_in_dict(text_ent[i], entity_to_idx):
                if i - idx > closest_distance:
                    print("Nekaj je narobe!!! Preiskujemo dlje kot že znana najbližja entiteta")
                if i - idx < closest_distance:
                    # zbrisi prejsnje najblizje
                    closest_entities = []
                for ent in text_ent[i]:
                    closest_entities.append(ent)
                break
            i += 1
        for entity in closest_entities:
            if not isinstance(entity, int):
                print('Kako da to ni numeric?')
            elif entity not in entity_to_idx:
                # print("Entiteta {} ni v entity to idx listi?!?!?".format(entity))
                pass
            else:
                X[entity_to_idx[entity], column_in_x] += 1
        if len(closest_entities) == 0:
            print("Mater ejga!!. Vsaj ena entiteta mora biti najbližja !?!?")
    return X


def get_single_entity_in_sentence(X, dataset, sentiment_arr, entity_to_idx, pos_column_in_x=4, neg_column_in_x=5):
    """
        If a single entity is in one sentence, then add all sentiments that are in a sentance to that entity.
        Sentence is considered as words between '.' or ','!
    :param X: write result here
    :param dataset: original dataset of a text
    :param sentiment_arr: sentiment array of text
    :param entity_to_idx: map from entity index to index in X row
    :param pos_column_in_x: X column that indicates positive sentiment
    :param neg_column_in_x: X column that indicates negative sentiment
    :return: X
    """
    start_idx = 0
    curr_entity = None
    pos_sentiments = 0
    neg_sentiments = 0
    last_sentiment_idx = 0
    go_to_next_sentence = False

    for idx, (token, entities) in enumerate(dataset):
        if go_to_next_sentence:
            if token == '.' or token == ',':
                curr_entity = None
                pos_sentiments = 0
                neg_sentiments = 0
                go_to_next_sentence = False
            continue

        while last_sentiment_idx < len(sentiment_arr) - 1 and sentiment_arr[last_sentiment_idx][0] < idx:
            last_sentiment_idx += 1

        if sentiment_arr[last_sentiment_idx][0] == idx:
            if sentiment_arr[last_sentiment_idx][1] == 1:
                pos_sentiments += 1
            else:
                neg_sentiments += 1

        if len(entities) > 0 and entities[0] in entity_to_idx:
            if curr_entity is None:
                curr_entity = entities[0]
            elif curr_entity != entities[0]:
                # go to next sentence
                go_to_next_sentence = True

        if token == '.' or token == ',':
            if curr_entity in entity_to_idx:
                X[entity_to_idx[curr_entity], pos_column_in_x] += pos_sentiments
                X[entity_to_idx[curr_entity], neg_column_in_x] += neg_sentiments
            curr_entity = None
            pos_sentiments = 0
            neg_sentiments = 0
            go_to_next_sentence = False
    return X


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
