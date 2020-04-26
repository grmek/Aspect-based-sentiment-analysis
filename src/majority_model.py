from general_model import GeneralModel


class MajorityModel(GeneralModel):
    def fit(self, dataset, sentiment_dicts):
        """
            Model fitting does nothing in this case
        :param dataset: list of data
        :param sentiment_dicts: list of ground truth dictionaries
        """
        pass

    def predict(self, dataset, sentiment_dicts):
        """
            Set all predicted values to majority class 3
        :param dataset: list of data
        :param sentiment_dicts: list of entity dictionaries
        :return: predicted sentiments (list of dictionaries)
        """
        result = []
        for sen_dict in sentiment_dicts:
            d = dict()
            for entity in sen_dict.keys():
                d[entity] = 3
            result.append(d)
        return result
