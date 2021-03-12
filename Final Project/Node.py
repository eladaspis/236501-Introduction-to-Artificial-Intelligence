class Node:
    def __init__(self, feature=None, left=None, right=None, classification=None, threshold=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.classification = classification
        self.threshold = threshold

    def dt_classify(self, features, example):
        if self.left is None and self.right is None:
            return self.classification
        if float(example[features.index(self.feature)]) < self.threshold:
            return self.left.dt_classify(features, example)
        else:
            return self.right.dt_classify(features, example)


class ImprovedNode(Node):
    def __init__(self, feature=None, left=None, right=None, classification=None, threshold=None,
                 majority_classification_of_examples=None, delta=None):
        Node.__init__(self, feature, left, right, classification, threshold)
        self.majority_classification_of_examples = majority_classification_of_examples
        self.delta = delta
        self.second_threshold = None
        self.middle = None

    def dt_classify(self, features, example):
        if self.left is None and self.right is None:
            return self.classification
        if float(example[features.index(self.feature)]) < self.threshold:
            return self.left.DT_Classify(features, example)
        else:
            return self.right.DT_Classify(features, example)

    def dt_classify_improved(self, features, example, eps=None):
        if self.left is None and self.right is None and self.middle is None:
            return self.classification, self.majority_classification_of_examples
        if self.middle is not None:
            right_classification, right_majority = self.right.dt_classify_improved(features, example, eps)
            left_classification, left_majority = self.left.dt_classify_improved(features, example, eps)
            middle_classification, middle_majority = self.middle.dt_classify_improved(features, example, eps)
            if float(example[features.index(self.feature)]) < self.threshold:
                if eps is not None and abs(
                        float(example[features.index(self.feature)]) - self.threshold) < eps * self.threshold:
                    if left_majority > middle_majority:
                        return left_classification, left_majority
                    else:
                        return middle_classification, middle_majority
                return left_classification, left_majority
            elif self.threshold <= float(example[features.index(self.feature)]) < self.second_threshold:
                if eps is not None and abs(
                        float(example[features.index(self.feature)]) - self.threshold) < eps * self.threshold:
                    if left_majority > middle_majority:
                        return left_classification, left_majority
                    else:
                        return middle_classification, middle_majority
                if eps is not None and abs(float(
                        example[features.index(self.feature)]) - self.second_threshold) <= eps * self.second_threshold:
                    if right_majority > middle_majority:
                        return right_classification, right_majority
                    else:
                        return middle_classification, middle_majority
                return middle_classification, middle_majority
            else:
                if eps is not None and abs(float(
                        example[features.index(self.feature)]) - self.second_threshold) <= eps * self.second_threshold:
                    if right_majority > middle_majority:
                        return right_classification, right_majority
                    else:
                        return middle_classification, middle_majority
                return right_classification, right_majority
        else:  # right and left only exist and that's why self.second_threshold is None
            if eps is not None and abs(
                    float(example[features.index(self.feature)]) - self.threshold) < eps * self.threshold:
                right_classification, right_majority = self.right.dt_classify_improved(features, example, eps)
                left_classification, left_majority = self.left.dt_classify_improved(features, example, eps)
                if right_majority > left_majority:
                    return right_classification, right_majority
                else:
                    return left_classification, left_majority
            elif float(example[features.index(self.feature)]) < self.threshold:
                return self.left.dt_classify_improved(features, example, eps)
            else:
                return self.right.dt_classify_improved(features, example, eps)
