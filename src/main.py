import os
from evaluation import Evaluation
from random_model import RandomModel
from majority_model import MajorityModel
from lexicon_features_model import LexiconFeaturesModel
from bert_model import BertModel


DATA_DIR = '../data/'
DATASET_DIR = DATA_DIR + 'SentiCoref_1.0'
SENTIMENT_LEXICON_DIR = DATA_DIR + 'Lexicon'


def main():
    """
        MAIN OHOHOHOHOHO
    """
    sentiment_lexicon_dict = get_sentiment_lexicon(SENTIMENT_LEXICON_DIR)

    evaluation = Evaluation(DATASET_DIR)

    # RandomModel
    evaluation.evaluate(RandomModel())

    # MajorityModel
    evaluation.evaluate(MajorityModel())

    # LexiconFeaturesModel
    evaluation.evaluate(LexiconFeaturesModel(sentiment_lexicon_dict))
    evaluation.evaluate(LexiconFeaturesModel(sentiment_lexicon_dict, positive_around_num=[3, 5, 10],
                                             negative_around_num=[3, 5, 10], normalize_data=True))
    evaluation.evaluate(LexiconFeaturesModel(sentiment_lexicon_dict, positive_around_num=[3, 5, 10],
                                             negative_around_num=[3], normalize_data=True))
    evaluation.evaluate(LexiconFeaturesModel(sentiment_lexicon_dict, positive_around_num=[5],
                                             negative_around_num=[3, 5, 10], normalize_data=True))
    evaluation.evaluate(LexiconFeaturesModel(sentiment_lexicon_dict, positive_around_num=[3, 5, 10, 15],
                                             negative_around_num=[3, 5, 10], normalize_data=False))
    evaluation.evaluate(LexiconFeaturesModel(sentiment_lexicon_dict, positive_around_num=[3, 15],
                                             negative_around_num=[10, 15], normalize_data=False))

    # BertModel
    evaluation.evaluate(BertModel(n_words_left_right=1, conv_filters=100, dense_units=256,
                                  dropout_rate=0.2, batch_size=128, epochs=5))
    evaluation.evaluate(BertModel(n_words_left_right=2, conv_filters=100, dense_units=256,
                                  dropout_rate=0.2, batch_size=128, epochs=5))
    evaluation.evaluate(BertModel(n_words_left_right=3, conv_filters=100, dense_units=256,
                                  dropout_rate=0.2, batch_size=128, epochs=5))
    evaluation.evaluate(BertModel(n_words_left_right=4, conv_filters=100, dense_units=256,
                                  dropout_rate=0.2, batch_size=128, epochs=5))
    evaluation.evaluate(BertModel(n_words_left_right=5, conv_filters=100, dense_units=256,
                                  dropout_rate=0.2, batch_size=128, epochs=5))
    evaluation.evaluate(BertModel(n_words_left_right=6, conv_filters=100, dense_units=256,
                                  dropout_rate=0.2, batch_size=128, epochs=5))
    evaluation.evaluate(BertModel(n_words_left_right=7, conv_filters=100, dense_units=256,
                                  dropout_rate=0.2, batch_size=128, epochs=5))


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


if __name__ == '__main__':
    main()
