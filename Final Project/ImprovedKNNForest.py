import math
import random
from CostSensitiveID3 import CostSensitiveID3
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


class ImprovedKNNForest(AbstractLearning):
    """
    ImprovedKNNForest algorithm
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
        self.minmax_value = None  # list of tuple (min_value, max_value)
        self.used_features = []

    def fit(self, examples, features):
        """
        ImprovedKNNForest algotithm
        input:
            examples - set of exmaples
            features - set of features
        output: save list of Improved TDIDT trees and compute centroids
        In addition, I change everytime the m_param and I got that m_param=1 give me the highest accuracy
        """
        self.examples = examples
        self.features = features
        self.minmax_value = [(float('inf'), float('-inf')) for i in range(len(self.features) - 1)]
        self.n_param = len(self.examples)
        for i in range(self.N_param):
            current_examples = random.choices(self.examples, k=int(self.p_param * self.n_param))
            current_tree = CostSensitiveID3(m_param=1)
            current_used_features = set()
            current_tree.fit(current_examples, self.features, current_used_features)
            self.used_features.append(current_used_features)
            current_centroid = centroid(current_examples)
            self.centroid_of_trees.append(current_centroid)
            self.trees.append(current_tree)

    def compute_eucliden_distance(self, x, y, current_used_features):
        assert len(x) == len(y)
        sum_value = 0
        for i in range(len(x)):
            if self.features[i] in current_used_features:
                sum_value += 2 * (math.pow(float(x[i]) - float(y[i]), 2))
            else:
                sum_value += math.pow(float(x[i]) - float(y[i]), 2)
        return math.sqrt(sum_value)

    def predict(self, test_examples, to_print=True, accuracy=False, loss=False):
        right = 0
        false_negative, false_positive = 0, 0
        for example in test_examples:
            positive, negative = 0, 0
            current_k_nearest_trees = []
            for tree_index in range(len(self.centroid_of_trees)):
                current_distance = self.compute_eucliden_distance(example[1:], self.centroid_of_trees[tree_index],
                                                                  self.used_features[tree_index])
                current_k_nearest_trees.append((current_distance, tree_index))
            current_k_nearest_trees.sort(key=lambda k: k[0])
            current_k_nearest_trees = current_k_nearest_trees[:self.k_param]
            for i in range(self.k_param):
                classification = (self.trees[current_k_nearest_trees[i][1]].root.dt_classify_improved(self.features,
                                                                                                      example,
                                                                                                      eps=0.1))[0]
                if classification == POSITIVE:
                    positive += (self.k_param - i)
                else:
                    negative += (self.k_param - i)
            current_classification = POSITIVE if positive > negative else NEGATIVE
            if current_classification == POSITIVE and current_classification is not example[0]:
                false_positive += 1
            if current_classification == NEGATIVE and current_classification is not example[0]:
                false_negative += 1
            if current_classification == example[0]:
                right += 1
        if accuracy:
            accuracy = right / len(test_examples)
            if to_print:
                print(accuracy)
            else:
                return accuracy
        if loss:
            loss = (0.1 * false_positive + false_negative) / len(test_examples)
            if to_print:
                print(loss)
            else:
                return loss

    def predict1(self, test_examples, to_print=True, accuracy=False, loss=False):
        right = 0
        false_negative, false_positive = 0, 0
        for example in test_examples:
            positive, negative = 0, 0
            current_k_nearest_trees = []
            for tree_index in range(len(self.centroid_of_trees)):
                current_distance = self.compute_eucliden_distance(example[1:], self.centroid_of_trees[tree_index],
                                                                  self.used_features[tree_index])
                current_k_nearest_trees.append((current_distance, tree_index))
            current_k_nearest_trees.sort(key=lambda k: k[0])
            current_k_nearest_trees = current_k_nearest_trees[:self.k_param + 1]
            for i in range(self.k_param):
                print("i: ", i)
                print(self.trees)
                classification = (self.trees[current_k_nearest_trees[i][1]].root.dt_classify_improved(self.features,
                                                                                                      example,
                                                                                                      eps=0.1))[0]
                if classification == POSITIVE:
                    positive += (self.k_param - i)
                else:
                    negative += (self.k_param - i)
            current_classification = POSITIVE if positive > negative else NEGATIVE
            if current_classification == POSITIVE and current_classification is not example[0]:
                false_positive += 1
            if current_classification == NEGATIVE and current_classification is not example[0]:
                false_negative += 1
            if current_classification == example[0]:
                right += 1
        if accuracy:
            accuracy = right / len(test_examples)
            if to_print:
                print(accuracy)
            else:
                return accuracy
        if loss:
            loss = (0.1 * false_positive + false_negative) / len(test_examples)
            if to_print:
                print(loss)
            else:
                return loss


if __name__ == '__main__':
    examples, features = load_data("train.csv")
    x_test, y_test = load_data("test.csv")

    # 7.2
    # expermient - each time with different with p_param
    # K = [21, 31, 41]
    # N = [50, 60, 70]
    # for i in K:
    #     for j in N:
    #         print("k: ", i, ", N: ", j)
    #         knn_improved = ImprovedKNNForest(p_param=0.7, N_param=j, k_param=i)
    #         knn_improved.fit(examples, features)
    #         knn_improved.predict(x_test, accuracy=True)

    knn_improved = ImprovedKNNForest(p_param=0.7, N_param=60, k_param=31)
    knn_improved.fit(examples, features)
    knn_improved.predict(x_test, accuracy=True)
