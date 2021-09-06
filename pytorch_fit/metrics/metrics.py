from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Metric(ABC):
    @abstractmethod
    def reset_state(self):
        ...

    @abstractmethod
    def update_state(self, y_pred, y_true, *args):
        ...

    @abstractmethod
    def value(self):
        ...

    @abstractmethod
    def is_value_simple(self):
        ...

    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def short_name(self):
        ...


class Accuracy(Metric):
    def __init__(self, is_correct):
        self.is_correct = is_correct
        self.reset_state()

    def reset_state(self):
        self.corrects = 0
        self.all_samples = 0

    def update_state(self, y_pred, y_true, *args):
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        results = self.is_correct(y_pred, y_true)

        self.corrects += np.sum(results)
        self.all_samples += len(results)

    def value(self):
        return self.corrects / self.all_samples

    def is_value_simple(self):
        return True

    def name(self):
        return "Accuracy"

    def short_name(self):
        return "acc"


class ConfusionMatrix(Metric):
    def __init__(self, all_labels, get_labels):
        self.all_labels = all_labels
        self.get_labels = get_labels
        self.reset_state()

    def reset_state(self):
        self.matrix = pd.DataFrame(
            data=0,
            columns=pd.Index(self.all_labels, name="actual:"),
            index=pd.Index(self.all_labels, name="predicted:"),
        )

    def update_state(self, y_pred, y_true, *args):
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        preds_labels = self.get_labels(y_pred)
        actual_labels = self.get_labels(y_true)

        for pred_label, actual_label in zip(preds_labels, actual_labels):
            self.matrix.loc[pred_label, actual_label] += 1

    def value(self):
        return self.matrix

    def is_value_simple(self):
        return False

    def name(self):
        return "Confusion"

    def short_name(self):
        return "confusion"
