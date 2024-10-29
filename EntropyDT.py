import numpy as np

from src.DecisionTree import DecisionTree, Node


class EntropyDT(DecisionTree):
    def __init__(self, X, y, min_branch):
        super().__init__(X, y, min_branch)

    def impurity(self, X: np.array, y: np.array) -> float:
        if np.unique(y).shape[0] == 1:
            return 0
        res = 0
        for k in range(y.max() + 1):
            p = (y == k).sum() / (y.shape[0])
            res -= p * np.log(p)
        return res

def print_node(node, depth=0):
    s = '\t' * depth
    if type(node) == Node:
        print(f"{s}Node:"
              f"{s}\tsep_i={node.feature_split_id}"
              f"{s}\tsep_v={node.split_value}")
        print_node(node.left_child, depth+1)
        print_node(node.right_child, depth+1)
    else:
        print(f"{s}Value: {int(node)}")


if __name__ == '__main__':
    N = 100
    r = 10

    X = np.random.randint(-r, r, size=(N, 2))
    Y = ((X[:, 0] < 0) == (X[:, 1] < 0)).astype(int)
    dtree = EntropyDT(X, Y, 0.1)
    dtree.train()


    print_node(dtree.root_node)

    print(dtree.predict(X))