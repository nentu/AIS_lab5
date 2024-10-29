import numpy as np


class MetricManager:
    def __init__(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred

        self._cm = [[0, 0], [0, 0]]

        for pred in range(2):
            for real in range(2):
                self._cm[pred][real] = self.get_cell(pred, real)

        self.tp = self._cm[0][0]
        self.fn = self._cm[0][1]
        self.fp = self._cm[1][0]
        self.tn = self._cm[1][1]

        self.acc = self._get_acc()
        self.precision = self._get_precision()
        self.recall = self._get_recall()
        self.fpr = self._get_fpr()
    def get_cell(self, is_pred, is_real):
        return np.all([self.y_pred == is_pred, self.y_true == is_real], axis=0).sum()

    def print_cm(self):
        print('Predicted')
        print('P\tN')
        for row, text in zip(self._cm, ["True Positive", "False Positive"]):
            for cell in row:
                print(cell, end='\t')
            print(text)

    def _get_acc(self):
        return (self.tp + self.tn) / (np.sum(np.array(self._cm)))

    def _get_precision(self):
        return self.tp / (self.tp + self.fp)

    def _get_recall(self):
        return self.tp / (self.tp + self.fn)

    def _get_fpr(self):
        return self.fp / (self.fp + self.tn)

    def f1_score(self, beta=0.5):
        return (beta ** 2 + 1) * (self.recall * self.precision) / (self.recall + beta ** 2 * self.precision)

