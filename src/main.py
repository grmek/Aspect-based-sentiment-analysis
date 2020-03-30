import os
import random


def main():
    data_set_dir = '../data/SentiCoref_1.0'

    total_deviation = 0
    num_comparisons = 0

    for file_name in os.listdir(data_set_dir):
        data_sample = get_data_sample(data_set_dir + '/' + file_name)
        ground_truth = get_ground_truth(data_sample)
        method_1_results = get_sentiment_method_1(data_sample)
        avg_deviation = get_avg_deviation(ground_truth, method_1_results)

        total_deviation += avg_deviation * len(ground_truth)
        num_comparisons += len(ground_truth)

        print(ground_truth)
        print(avg_deviation)
        print()

    print(total_deviation / num_comparisons)


def get_data_sample(file_path):
    with open(file_path, encoding='utf-8') as f:
        return [line[:-2].split('\t') for line in f.readlines()[7:]]


def get_ground_truth(data_sample):
    sentiment_dict = dict()

    for entry in data_sample:
        if entry[-3] != '_':
            for entity, sentiment in zip(entry[-1].split('|'), entry[-3].split('|')):
                if entity != '_':
                    sentiment_dict[int(entity[2:-1])] = int(sentiment[0])

    return sentiment_dict


def get_avg_deviation(sentiment_dict_1, sentiment_dict_2):
    total_deviation = 0
    num_comparisons = 0

    for entity, sentiment in sentiment_dict_1.items():
        total_deviation += abs(sentiment_dict_2[entity] - sentiment)
        num_comparisons += 1

    return total_deviation / num_comparisons


def get_sentiment_method_1(data_sample):
    sentiment_dict = dict()

    for entry in data_sample:
        if entry[-3] != '_':
            for entity in entry[-1].split('|'):
                if entity != '_':
                    sentiment_dict[int(entity[2:-1])] = random.randint(1, 5)

    return sentiment_dict


if __name__ == '__main__':
    main()
