import math

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class C45:
    def __init__(self, attrNames):
        self.attrNames = attrNames

    def entropy(self, target):
        # Calculate entropy of the target attribute
        entropy = 0.0
        for label in np.unique(target):
            p = target[target == label].shape[0] / target.shape[0]
            entropy -= p * math.log2(p)
        return entropy

    def information_gain(self, X, y, attr_index):
        # Calculate information gain for an attribute
        gain = self.entropy(y)
        for value in np.unique(X[:, attr_index]):
            subset_y = y[X[:, attr_index] == value]
            gain -= (subset_y.shape[0] / y.shape[0]) * self.entropy(subset_y)
        return gain

    def split_info(self, X, attr_index):
        # Calculate split information for an attribute
        split_info = 0.0
        for value in np.unique(X[:, attr_index]):
            p = X[X[:, attr_index] == value].shape[0] / X.shape[0]
            split_info -= p * math.log2(p)
        return split_info

    def gain_ratio(self, X, y, attr_index):
        # Calculate gain ratio for an attribute
        gain = self.information_gain(X, y, attr_index)
        split_info = self.split_info(X, attr_index)
        if split_info == 0:
            return 0
        return gain / split_info

    def build_tree(self, X, y):
        # Build the decision tree using C4.5 algorithm
        if len(np.unique(y)) == 1:
            return y[0]  # Leaf node
        best_attr = None
        best_gain_ratio = -1
        for i in range(X.shape[1]):
            gain_ratio = self.gain_ratio(X, y, i)
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_attr = i
        # Split the data based on the best attribute
        values = np.unique(X[:, best_attr])
        children = dict()
        for value in values:
            subset_X = X[X[:, best_attr] == value]
            subset_y = y[X[:, best_attr] == value]
            children[value] = self.build_tree(subset_X, subset_y)
        return {best_attr: children}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def print_tree(self):
        # Print the decision tree
        def print_tree_recursive(tree, indent=0):
            if isinstance(tree, dict):
                for key, value in tree.items():
                    print(' ' * indent + self.attrNames[key] + ':')
                    print_tree_recursive(value, indent + 2)
            else:
                print(' ' * indent + str(tree))

        print_tree_recursive(self.tree)

    def get_leaf(self, x, node = None):
        if node is None:
            node = self.tree

        if isinstance(node, dict):
            split_att = list(node.keys())[0]
            return self.get_leaf(x, node[split_att][x[split_att]])
        else:
            return node


    def predict(self, X):
        res = list()
        for x in X:
            res.append(
                self.get_leaf(x)
            )

        return np.array(res)




if __name__ == '__main__':
    # Example usage
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)
    clf = C45(iris.feature_names)
    clf.fit(X_train, y_train)
    clf.print_tree()
