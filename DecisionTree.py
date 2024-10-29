from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    feature_split_id: int
    split_value: float
    left_child = None
    right_child = None


@dataclass
class BestSplit:
    feature_split_id: int
    split_v: float
    branch: float


def _sort_by_feat_id(X: np.array, y: np.array, feat_id) -> np.array:
    # Get the indices that would sort the array based on the second column
    indices = np.argsort(X[:, feat_id])

    # Use these indices to sort the array
    return X[indices], y[indices]


def most_frequent(array: np.array) -> float:
    counts = np.bincount(array)
    res = np.argmax(counts)
    return res


class DecisionTree:
    def __init__(self, X: np.array, y: np.array, min_branch) -> None:
        self.min_branch = min_branch
        self.y = y
        self.root_node = None
        self.X = X

    def impurity(self, X: np.array, y: np.array) -> float:
        pass

    def _get_value_i_list(self, X):  # histogram
        return np.unique(X)[:-1]

    def create_node(self, X: np.array, y: np.array, depth):
        h = self.impurity(X, y)
        if h == 0:
            return y[0]
        best_split = None
        for feat_id in range(X.shape[1]):
            for split_v in self._get_value_i_list(X[:, feat_id]):
                left_idx = X[:, feat_id] <= split_v
                right_idx = ~left_idx

                assert ((left_idx + right_idx) == [1] * X.shape[0]).all()

                left_h = self.impurity(X[left_idx], y[left_idx])
                right_h = self.impurity(X[right_idx], y[right_idx])
                branch = X.shape[0] * h - left_h * left_idx.sum() - right_h * right_idx.sum()
                if best_split is None or branch >= best_split.branch:
                    if best_split is None:
                        best_split = BestSplit(feat_id, split_v, branch)
                    else:
                        best_split.feature_split_id = feat_id
                        best_split.split_v = split_v
                        best_split.branch = branch

        new_node = Node(
            best_split.feature_split_id,
            best_split.split_v
        )

        left_idx = X[:, best_split.feature_split_id] <= best_split.split_v
        right_idx = X[:, best_split.feature_split_id] > best_split.split_v

        if depth == 0 or best_split.branch < self.min_branch:

            left_y = y[left_idx]
            right_y = y[right_idx]
            new_node.left_child = most_frequent(left_y)
            new_node.right_child = most_frequent(right_y)

        else:
            new_node.left_child = self.create_node(
                X[left_idx], y[left_idx],
                depth - 1
            )
            new_node.right_child = self.create_node(
                X[right_idx], y[right_idx],
                depth - 1
            )
        return new_node

    def train(self, depth=10):
        self.root_node = self.create_node(self.X, self.y, depth)

    def _go_to_leaf(self, x, cur_node: Node):
        if type(cur_node) != Node:
            return cur_node

        if x[cur_node.feature_split_id] <= cur_node.split_value:
            return self._go_to_leaf(x, cur_node.left_child)
        return self._go_to_leaf(x, cur_node.right_child)

    def predict(self, X):
        assert self.root_node is not None
        y_list = list()
        for x in X:
            y = self._go_to_leaf(x, self.root_node)
            y_list.append(y)

        return np.array(y_list)
