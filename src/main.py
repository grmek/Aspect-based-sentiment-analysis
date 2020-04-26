import os
from evaluation import Evaluation
from random_model import RandomModel
from majority_model import MajorityModel
from lexicon_features_model import LexiconFeaturesModel


DATA_DIR = '../data/'
DATASET_DIR = DATA_DIR + 'SentiCoref_1.0'
SENTIMENT_LEXICON_DIR = DATA_DIR + 'Lexicon'


def main():
    """
        MAIN OHOHOHOHOHO
    """
    sentiment_lexicon_dict = get_sentiment_lexicon(SENTIMENT_LEXICON_DIR)

    evaluation = Evaluation(DATASET_DIR)

    evaluation.evaluate(RandomModel())
    evaluation.evaluate(MajorityModel())
    evaluation.evaluate(LexiconFeaturesModel(sentiment_lexicon_dict))


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
