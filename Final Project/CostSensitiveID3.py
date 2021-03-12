from sklearn.model_selection import KFold
from Node import ImprovedNode
from helper import AbstractLearning
from helper import load_data

POSITIVE = 'M'  # M=sick label
NEGATIVE = 'B'  # B=healthy label
PURE_POSITIVE = 'm'
PURE_NEGATIVE = 'b'
EMPTY = 1


class CostSensitiveID3(AbstractLearning):
    def __init__(self, m_param=None):
        AbstractLearning.__init__(self)
        self.m_param = m_param
        self.root = ImprovedNode()

    def dev_sons(self, current_threshold, examples, current_feature_index):
        left_examples, right_examples = [], []
        for example in examples:
            if float(example[current_feature_index]) <= current_threshold:
                left_examples.append(example)
            else:
                right_examples.append(example)
        return right_examples, left_examples

    def dev_sons_improved(self, first_threshold, second_threshold, examples, current_feature_index):
        left_examples, mid_examples, right_examples = [], [], []
        for example in examples:
            if second_threshold is not None:
                if float(example[current_feature_index]) <= first_threshold:
                    left_examples.append(example)
                elif float(example[current_feature_index]) < second_threshold:
                    mid_examples.append(example)
                else:
                    right_examples.append(example)
            else:
                if float(example[current_feature_index]) <= first_threshold:
                    left_examples.append(example)
                else:
                    right_examples.append(example)
        return right_examples, mid_examples, left_examples

    def TDIDT(self, examples, features, default_classifier, root, used_features):
        if len(examples) == 0:
            return None
        if self.m_param is not None and len(examples) < self.m_param:
            return ImprovedNode(classification=default_classifier, majority_classification_of_examples=len(examples))
        classifier = self.MajorityClass(examples)
        if classifier == PURE_NEGATIVE:
            return ImprovedNode(classification=NEGATIVE, majority_classification_of_examples=len(examples))
        elif classifier == PURE_POSITIVE:
            return ImprovedNode(classification=POSITIVE, majority_classification_of_examples=len(examples))
        current_feature_index, main_threshold, second_threshold = self.select_feature(examples, features, improved=True)
        current_feature = features[current_feature_index]
        mid_examples = []
        if second_threshold is None:
            right_examples, left_examples = self.dev_sons(main_threshold, examples, current_feature_index)
        else:
            right_examples, mid_examples, left_examples = self.dev_sons_improved(main_threshold, second_threshold,
                                                                                 examples, current_feature_index)
        if used_features is not None:
            used_features.add(current_feature)
        root.feature = current_feature
        root.threshold = main_threshold
        root.second_threshold = second_threshold
        if right_examples is not [] and left_examples is not [] and len(features) != 2:
            root.left = ImprovedNode()
            root.right = ImprovedNode()
            root.left = self.TDIDT(left_examples, features, classifier,
                                   root.left, used_features)
            if second_threshold is not None:
                root.middle = ImprovedNode()
                root.middle = self.TDIDT(mid_examples, features, classifier, root.middle, used_features)
            root.right = self.TDIDT(right_examples, features,
                                    classifier, root.right, used_features)
        return root

    def fit_improved(self, examples, features):
        self.features = features
        kf = KFold(n_splits=5, shuffle=True, random_state=313597023)
        m_parameters = [1, 2, 3, 4]
        minimum_loss = float('inf')
        for m_parameter in m_parameters:
            for train_index, test_index in kf.split(examples):
                train_list, test_list = [], []
                for i in train_index:
                    train_list.append(examples[i])
                for i in test_index:
                    test_list.append(examples[i])
                current_tree = CostSensitiveID3(m_param=m_parameter)
                current_tree.fit(train_list, features)
                current_loss = current_tree.predict(test_list, to_print=False, loss=True, improved=True, eps=0.1)
                if current_loss < minimum_loss:
                    self.root = current_tree.root
                    minimum_loss = current_loss

    def fit(self, examples, features, used_features=None):
        """
        ID3 algotithm
        input:
            examples - set of examples
            features - set of features
            used_features - in the end of the program that list will contain all the features that TDIDT split by
        output: Improved ID3 Tree
        """
        self.features = features
        classifier = self.MajorityClass(examples)
        self.root = self.TDIDT(examples, features, classifier, self.root, used_features)

    def predict(self, test_examples, to_print=True, accuracy=False, loss=False, eps=None, improved=False):
        num_of_input = len(test_examples)
        wrong, right, classification = 0, 0, None
        false_negative, false_positive = 0, 0
        for example in test_examples:
            if improved:
                classification = (self.root.dt_classify_improved(self.features, example, eps=eps))[0]
            else:
                classification = (self.root.dt_classify_improved(self.features, example))[0]

            if classification == POSITIVE and classification is not example[0]:
                false_positive += 1
            if classification == NEGATIVE and classification is not example[0]:
                false_negative += 1
            if classification == example[0]:
                right += 1
            else:
                wrong += 1
        if accuracy:
            accuracy = right / num_of_input
            if to_print:
                print(accuracy)
            else:
                return accuracy
        if loss:
            loss = (0.1 * false_positive + false_negative) / num_of_input
            if to_print:
                print(loss)
            else:
                return loss


if __name__ == '__main__':
    examples, features = load_data("train.csv")
    x_test, y_test = load_data("test.csv")

    # 4.1 - print loss value
    # for i in range(1, 30):
    # print("before: ")
    # algo = ID3(m_param=1)
    # algo.fit(examples, features)
    # algo.predict(x_test, loss=True, to_print=True)
    #
    # experiment
    # eps_list = np.linspace(0.05, 0.15, num=20)
    # for i in eps_list:
    #     print("eps: " i)
    #     algo_improved_id = CostSensitiveID3()
    #     algo_improved_id.fit_improved(examples, features)
    #     algo_improved_id.predict(x_test, loss=True, eps=i, improved=True)

    # 4.3 - print CostSensitiveID3
    # print("after: ")
    #
    algo_improved_id = CostSensitiveID3()
    algo_improved_id.fit_improved(examples, features)
    algo_improved_id.predict(x_test, loss=True, eps=0.07, improved=True)
