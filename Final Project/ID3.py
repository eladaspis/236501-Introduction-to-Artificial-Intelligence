import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from Node import Node
from helper import AbstractLearning
from helper import load_data

POSITIVE = 'M'  # M=sick label
NEGATIVE = 'B'  # B=healthy label
PURE_POSITIVE = 'm'
PURE_NEGATIVE = 'b'
EMPTY = 1


class ID3(AbstractLearning):
    """
    ID3 algorithm
    """

    def __init__(self, m_param=None):
        AbstractLearning.__init__(self)
        self.m_param = m_param
        self.root = Node()

    def TDIDT(self, examples, features, default_classifier, root):
        if len(examples) == 0:
            return None, None, default_classifier
        classifier = self.MajorityClass(examples)
        if self.m_param is not None and len(examples) < self.m_param:
            return Node(classification=default_classifier)
        if classifier == PURE_NEGATIVE:
            return Node(classification=NEGATIVE)
        elif classifier == PURE_POSITIVE:
            return Node(classification=POSITIVE)
        current_feature_index, current_threshold = (self.select_feature(examples, features))
        current_feature = features[current_feature_index]
        root.feature = current_feature
        right_examples, left_examples = self.dev_sons(current_threshold, examples, current_feature_index)
        root.threshold = current_threshold
        if right_examples is not [] and left_examples is not [] and len(features) != 2:
            root.left = Node()
            root.right = Node()
            root.left = self.TDIDT(left_examples, features, classifier,
                                   root.left)
            root.right = self.TDIDT(right_examples, features,
                                    classifier, root.right)
        return root

    def fit(self, examples, features):
        """
        ID3 algotithm
        input:
            examples - set of exmaples
            features - set of features
        output: ID3 Tree
        """
        self.features = features
        classifier = self.MajorityClass(examples)
        self.root = self.TDIDT(examples, features, classifier, self.root)

    def predict(self, test_examples, to_print=True, accuracy=False, loss=False):
        num_of_input = len(test_examples)
        wrong, right = 0, 0
        false_negative, false_positive = 0, 0
        for example in test_examples:
            classification = self.root.dt_classify(self.features, example)
            if classification == example[0]:
                right += 1
            else:
                wrong += 1
            if classification == POSITIVE and classification is not example[0]:
                false_positive += 1
            if classification == NEGATIVE and classification is not example[0]:
                false_negative += 1
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


def experiment(file_path):
    '''
    just uncomment the experiment in the main program
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=313597023)
    m_parameters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    examples, features = load_data(file_path)
    average_accuracy = []
    for m_parameter in m_parameters:
        accuracy_list = []
        for train_index, test_index in kf.split(examples):
            train_list, test_list = [], []
            for i in train_index:
                train_list.append(examples[i])
            for i in test_index:
                test_list.append(examples[i])
            classifier = ID3(m_param=m_parameter)
            classifier.fit(train_list, features)
            current_accuracy = classifier.predict(test_list, to_print=False, accuracy=True)
            accuracy_list.append(current_accuracy)
        average_accuracy.append(sum(accuracy_list) / len(accuracy_list))
    plt.plot(m_parameters, average_accuracy)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Minimum Number of Examples in a Node')
    plt.show()


if __name__ == '__main__':
    examples, features = load_data("train.csv")
    x_test, y_test = load_data("test.csv")

    algo = ID3()
    algo.fit(examples, features)

    # 1.1 - print accuracy value
    algo.predict(x_test, accuracy=True)

    # 3.3 - print accuracy value
    # experiment("train.csv")

    # 3.4 -
    for i in range(1, 10):
        algo = ID3(m_param=i)
        algo.fit(examples, features)
        algo.predict(x_test, accuracy=True, to_print=False)
