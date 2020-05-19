from general_model import GeneralModel
import os
import numpy as np
from transformers import BertTokenizer, TFBertPreTrainedModel, TFBertMainLayer
import tensorflow as tf
from tensorflow.keras import layers


class BertModel(GeneralModel):
    def __init__(self, n_words_left_right=3, conv_filters=100, dense_units=256,
                 dropout_rate=0.2, batch_size=128, epochs=5):
        self.n_words_left_right = n_words_left_right
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs

        param_str = '%d_%d_%d_%d_%d_%d' % (n_words_left_right, conv_filters, dense_units,
                                           int(dropout_rate * 100), batch_size, epochs)
        self.name = self.__class__.__name__ + '_' + param_str
        self.model_path = '../data/models/bert_' + param_str

        self.random = np.random.RandomState(123)

        self.model = None

    def get_name(self):
        return self.name

    def fit(self, dataset, sentiment_dicts):
        if not os.path.isdir(self.model_path):
            self.print('Fitting the model ...')

            X = []
            y = []

            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

            # For each file ...
            for file_data, file_sentiments in zip(dataset, sentiment_dicts):
                # Go through text and construct the samples for training.
                for data_idx in range(len(file_data)):
                    for entity in file_data[data_idx][1]:
                        if entity in file_sentiments:
                            # For each entity mention, we construct a learning sample: neighbouring words (encoded with
                            # bert tokenizer) and entity's sentiment (from 0 to 4).
                            start_idx = max(data_idx - self.n_words_left_right, 0)
                            end_idx = min(data_idx + self.n_words_left_right + 1, len(file_data))
                            neighbouring_words = ' '.join([d[0] for d in file_data[start_idx:end_idx]])
                            X.append(bert_tokenizer.encode_plus(neighbouring_words, add_special_tokens=True,
                                                                max_length=128, pad_to_max_length=True)['input_ids'])
                            y.append(file_sentiments[entity] - 1)

            train_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))).batch(self.batch_size)

            self.model = BertConvDenseNN.from_pretrained('bert-base-multilingual-uncased',
                                                         conv_filters=self.conv_filters,
                                                         dense_units=self.dense_units,
                                                         dropout_rate=self.dropout_rate)
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                               metrics=['sparse_categorical_accuracy'])
            self.model.fit(train_data, epochs=self.epochs)
            self.model.save_weights(self.model_path + '/model', save_format='tf')

        else:
            self.print('Loading the model ...')

            self.model = BertConvDenseNN.from_pretrained('bert-base-multilingual-uncased',
                                                         conv_filters=self.conv_filters,
                                                         dense_units=self.dense_units,
                                                         dropout_rate=self.dropout_rate)
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                               metrics=['sparse_categorical_accuracy'])
            self.model.train_on_batch(np.array([[0] * 128]), np.array([0]))
            self.model.load_weights(self.model_path + '/model')

    def predict(self, dataset, sentiment_dicts):
        self.print('Predicitng results ...')

        X = []
        y = []
        files = []
        entities = []

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        # For each file ...
        for i, (file_data, file_sentiments) in enumerate(zip(dataset, sentiment_dicts)):
            # Go through text and construct the samples for predicting.
            for data_idx in range(len(file_data)):
                for entity in file_data[data_idx][1]:
                    if entity in file_sentiments:
                        start_idx = max(data_idx - self.n_words_left_right, 0)
                        end_idx = min(data_idx + self.n_words_left_right + 1, len(file_data))
                        neighbouring_words = ' '.join([d[0] for d in file_data[start_idx:end_idx]])
                        X.append(bert_tokenizer.encode_plus(neighbouring_words, add_special_tokens=True,
                                                            max_length=128, pad_to_max_length=True)['input_ids'])
                        y.append(file_sentiments[entity] - 1)
                        files.append(i)
                        entities.append(entity)

        test_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))).batch(self.batch_size)

        # The predictions are done for all samples individually, then we sum over all predictions for each entity, and
        # finally we determine the maximum class response (most probable sentiment) for each entity.
        predictions = self.model.predict(test_data)

        new_sentiment_dicts = [{k: np.zeros(5) for k in s.keys()} for s in sentiment_dicts]

        for i in range(predictions.shape[0]):
            new_sentiment_dicts[files[i]][entities[i]] += predictions[i]

        for i in range(len(new_sentiment_dicts)):
            for k, v in new_sentiment_dicts[i].items():
                new_sentiment_dicts[i][k] = np.argmax(v) + 1

        return new_sentiment_dicts


class BertConvDenseNN(TFBertPreTrainedModel):
    def __init__(self, config, conv_filters=100, dense_units=256, dropout_rate=0.2, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name='bert', trainable=False)
        self.conv_1 = layers.Conv1D(filters=conv_filters, kernel_size=2, padding='valid', activation='relu')
        self.conv_2 = layers.Conv1D(filters=conv_filters, kernel_size=3, padding='valid', activation='relu')
        self.conv_3 = layers.Conv1D(filters=conv_filters, kernel_size=4, padding='valid', activation='relu')
        self.pool = layers.GlobalMaxPool1D()
        self.dense_1 = layers.Dense(units=dense_units, activation='relu')
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.dense_2 = layers.Dense(units=5, activation='softmax')

    def call(self, inputs, training=False, **kwargs):
        bert = self.bert(inputs, training=training, **kwargs)
        conv_1 = self.conv_1(bert[0])
        conv_1 = self.pool(conv_1)
        conv_2 = self.conv_2(bert[0])
        conv_2 = self.pool(conv_2)
        conv_3 = self.conv_3(bert[0])
        conv_3 = self.pool(conv_3)
        conv = tf.concat([conv_1, conv_2, conv_3], axis=-1)
        dense_1 = self.dense_1(conv)
        dropout = self.dropout(dense_1, training)
        dense_2 = self.dense_2(dropout)
        return dense_2
