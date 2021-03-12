import csv
from math import log

POSITIVE = 'M'  # M=sick label
NEGATIVE = 'B'  # B=healthy label
PURE_POSITIVE = 'm'
PURE_NEGATIVE = 'b'
EMPTY = 1


def load_data(path):
    test_examples = []
    with open(path, 'r') as File:
        csv_reader = csv.reader(File)
        test_examples.append(list(csv_reader))
    features = test_examples[0][0]
    test_examples[0].remove(features)
    test_examples = test_examples[0]
    return test_examples, features


class AbstractLearning:
    def __init__(self, ):
        self.features = None
        self.root = None

    def Entropy(self, data):
        """
        input:
            X - random variable with x(1),x(2),...,x(n) values
        output:
            return the entropy of X
        """
        positive, negative = 0, 0
        for example in data:
            if example[0] == NEGATIVE:
                negative += 1
            else:
                positive += 1
        length = len(data)
        probability_negative = (negative / length) if length != 0 else 0
        probability_positive = (positive / length) if length != 0 else 0
        if probability_negative == 0 or probability_positive == 0:
            return 0
        return -probability_negative * log(probability_negative, 2) - probability_positive * log(probability_positive,
                                                                                                 2)

    def MajorityClass(self, examples):
        positive = 0
        negative = 0
        for example in examples:
            if example[0] == NEGATIVE:
                negative += 1
            elif example[0] == POSITIVE:
                positive += 1
        if negative == 0:
            if positive > 0:
                return PURE_POSITIVE
            else:
                return EMPTY
        else:
            if positive == 0:
                return PURE_NEGATIVE
            else:
                return NEGATIVE if negative > positive else POSITIVE

    def information_gain(self, examples, feature):
        threshold, ig = 0, 0
        result_information_gain = float('-inf')
        examples.sort(key=lambda x: x[feature])
        curr_entropy = self.Entropy(examples)
        result_threshold = 0
        for i in range(len(examples) - 1):
            right_poss, left_poss = [], []
            threshold = (float(examples[i + 1][feature]) + float(examples[i][feature])) / 2;
            for example in examples:
                if float(example[feature]) < threshold:
                    left_poss.append(example)
                else:
                    right_poss.append(example)
            right_entropy = self.Entropy(right_poss)
            left_entropy = self.Entropy(left_poss)
            current_information_gain = curr_entropy - (len(right_poss) / (len(examples))) * right_entropy - \
                                       (len(left_poss) / (len(examples))) * left_entropy
            if result_information_gain < current_information_gain:
                result_information_gain = current_information_gain
                result_threshold = threshold
        return result_information_gain, result_threshold

    def add_left_right(self, examples, feature, threshold):
        left_poss, right_poss = [], []
        for example in examples:
            if float(example[feature]) < threshold:
                left_poss.append(example)
            else:
                right_poss.append(example)
        return right_poss, left_poss

    def information_gain_improved(self, examples, feature):
        threshold, chosen_result_threshold, ig = 0, 0, None
        best_right_poss, best_left_poss = [], []
        right_entropy, left_entropy, best_right_entropy, best_left_entropy = 0, 0, 0, 0
        result_information_gain, chosen_result_information_gain = float('-inf'), float('-inf')
        examples.sort(key=lambda x: x[feature])
        curr_entropy = self.Entropy(examples)
        result_threshold = 0
        for i in range(len(examples) - 1):
            threshold = (float(examples[i + 1][feature]) + float(examples[i][feature])) / 2
            right_poss, left_poss = self.add_left_right(examples, feature, threshold)
            right_entropy = self.Entropy(right_poss)
            left_entropy = self.Entropy(left_poss)
            current_information_gain = curr_entropy - (len(right_poss) / (len(examples))) * right_entropy - \
                                       (len(left_poss) / (len(examples))) * left_entropy
            if result_information_gain < current_information_gain:
                result_information_gain = current_information_gain
                result_threshold = threshold
                best_right_entropy = right_entropy
                best_left_entropy = left_entropy
                best_right_poss = right_poss
                best_left_poss = left_poss
        if best_right_entropy < best_left_entropy:
            chosen_poss = best_left_poss
        elif best_right_entropy > best_left_entropy:
            chosen_poss = best_right_poss
        elif best_right_entropy == best_left_entropy == 0:
            return result_information_gain, result_threshold, None
        else:
            chosen_poss = best_right_poss

        for chosen_index in range(len(chosen_poss) - 1):
            chosen_left, chosen_right = [], []
            chosen_threshold = (float(chosen_poss[chosen_index + 1][feature]) + float(
                chosen_poss[chosen_index][feature])) / 2
            for chosen_example in chosen_poss:
                if float(chosen_example[feature]) < chosen_threshold:
                    chosen_left.append(chosen_example)
                else:
                    chosen_right.append(chosen_example)
            chosen_left_entropy = self.Entropy(chosen_left)
            chosen_right_entropy = self.Entropy(chosen_right)
            chosen_current_information_gain = curr_entropy - (
                    len(chosen_left) / (len(chosen_poss))) * chosen_left_entropy - \
                                              (len(chosen_right) / (len(chosen_poss))) * chosen_right_entropy
            if chosen_result_information_gain < chosen_current_information_gain:
                chosen_result_information_gain = chosen_current_information_gain
                chosen_result_threshold = chosen_threshold
        return result_information_gain, result_threshold, chosen_result_threshold

    def select_feature(self, examples, features, improved=False):
        best_gain, best_first_threshold, best_second_threshold = float('-inf'), None, None
        best_feature = None
        for feature in range(1, len(features)):  # without column 1
            if improved:
                current_gain, current_threshold, second_threshold = self.information_gain_improved(examples, feature)
                if current_gain >= best_gain:
                    best_gain = current_gain
                    best_feature = feature
                    best_first_threshold = current_threshold
                    best_second_threshold = second_threshold
            else:
                current_gain, current_threshold = self.information_gain(examples, feature)
                if current_gain >= best_gain:
                    best_gain = current_gain
                    best_feature = feature
                    best_first_threshold = current_threshold
        if improved:
            if best_second_threshold is not None:
                if best_first_threshold < best_second_threshold:
                    return best_feature, best_first_threshold, best_second_threshold
                else:
                    return best_feature, best_second_threshold, best_first_threshold
            else:
                return best_feature, best_first_threshold, None

        return best_feature, best_first_threshold

    def dev_sons(self, current_threshold, examples, current_feature_index):
        left_examples, right_examples = [], []
        for example in examples:
            if float(example[current_feature_index]) < current_threshold:
                left_examples.append(example)
            else:
                right_examples.append(example)
        return right_examples, left_examples
