from general_model import GeneralModel
import random


class RandomModel(GeneralModel):
    def fit(self, dataset, sentiment_dicts):
        """
            Model fitting does nothing in this case
        :param dataset: list of data
        :param sentiment_dicts: list of ground truth dictionaries
        """
        pass

    def predict(self, dataset, sentiment_dicts):
        """
            Set all predicted values randomly
        :param dataset: list of data
        :param sentiment_dicts: list of entity dictionaries
        :return: predicted sentiments (list of dictionaries)
        """
        random.seed(123)

        result = []
        for sen_dict in sentiment_dicts:
            d = dict()
            for entity in sen_dict.keys():
                d[entity] = random.randint(1, 5)
            result.append(d)
        return result
