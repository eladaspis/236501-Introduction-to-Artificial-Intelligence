import math
import random

from ID3 import ID3
from helper import AbstractLearning
from helper import load_data

POSITIVE = 'M'  # M=sick label
NEGATIVE = 'B'  # B=healthy label
PURE_POSITIVE = 'm'
PURE_NEGATIVE = 'b'
EMPTY = 1


def eucliden_distance(x, y):
    assert len(x) == len(y)
    sum_value = 0
    for i in range(len(x)):
        sum_value += math.pow((float(x[i]) - y[i]), 2)
    return math.sqrt(sum_value)


def centroid(examples):
    centroid_value = []
    for i in range(1, len(examples[0])):
        current_sum = 0
        for example in examples:
            current_sum += float(example[i])
        centroid_value.append(current_sum / len(examples))
    return centroid_value


class KNNForest(AbstractLearning):
    """
    KNNForest algorithm
    """

    def __init__(self, k_param=None, p_param=None, N_param=None):
        AbstractLearning.__init__(self)
        self.examples = None
        self.features = None
        self.k_param = k_param
        self.p_param = p_param  # probability
        self.n_param = None
        self.N_param = N_param
        self.centroid_of_trees, self.trees = [], []

    def fit(self, examples, features):
        """
        KNNForest algotithm
        input:
            examples - set of exmaples
            features - set of features
        output: save data
        """
        self.examples = examples
        self.features = features
        self.n_param = len(self.examples)

        for i in range(self.N_param):
            current_examples = random.choices(self.examples, k=int(self.p_param * self.n_param))
            current_tree = ID3(m_param=3)
            current_tree.fit(current_examples, self.features)
            current_centroid = centroid(current_examples)
            self.centroid_of_trees.append(current_centroid)
            self.trees.append(current_tree)

    def predict(self, test_examples):
        right = 0
        for example in test_examples:
            positive, negative = 0, 0
            current_k_nearest_trees = []
            for tree_index in range(len(self.centroid_of_trees)):
                current_distance = eucliden_distance(example[1:], self.centroid_of_trees[tree_index])
                current_k_nearest_trees.append((current_distance, tree_index))
            current_k_nearest_trees.sort(key=lambda k: k[0])
            current_k_nearest_trees = current_k_nearest_trees[:self.k_param]
            for i in range(self.k_param):
                classification = self.trees[current_k_nearest_trees[i][1]].root.dt_classify(self.features, example)
                if classification == POSITIVE:
                    positive += 1
                else:
                    negative += 1
            current_classification = POSITIVE if positive > negative else NEGATIVE
            if current_classification == example[0]:
                right += 1
        accuracy = right / len(test_examples)
        print(accuracy)


if __name__ == '__main__':
    examples, features = load_data("train.csv")
    x_test, y_test = load_data("test.csv")

    # experiment - each time with different p_param
    # K = [31, 41, 51]
    # N = [52, 60, 70]
    # for i in K:
    #     for j in N:
    #             print("k ", i, ", N :", j)
    #             algo2 = KNNForest(p_param=0.7, N_param=j, k_param=i)
    #             algo2.fit(examples, features)
    #             algo2.predict(x_test)
    # 6.1
    algo2 = KNNForest(p_param=0.7, N_param=70, k_param=41)
    algo2.fit(examples, features)
    algo2.predict(x_test)
