import numpy as np


class EarlyStopping(object):
    def __init__(
            self,
            metric_category,
            metric_name,
            n_epochs=5,
            stats_fun=np.mean,
            min_change=-0.001,
            relative_change_type=False,
    ):
        self.metric_category = metric_category
        self.metric_name = metric_name
        self.n_epochs = n_epochs
        self.stats_fun = stats_fun
        self.min_change = min_change
        self.relative_change_type = relative_change_type

    def __call__(self, history):
        if history == {}:
            return False
        values = history[self.metric_category][self.metric_name]
        if len(values) <= self.n_epochs:
            return False
        else:
            extreme_value_to_compare = self.stats_fun(
                values[-self.n_epochs: -1])
            diff = values[-1] - extreme_value_to_compare
            if self.relative_change_type:
                compare_to_change = diff / np.abs(extreme_value_to_compare)
            else:
                compare_to_change = diff
            if self.min_change < 0:
                return compare_to_change > self.min_change
            else:
                return compare_to_change < self.min_change

    def or_(self, another_stopping):
        return EarlyStoppingsCombination(self, another_stopping, "or")

    def and_(self, another_stopping):
        return EarlyStoppingsCombination(self, another_stopping, "and")


class EarlyStoppingsCombination(EarlyStopping):
    def __init__(self, early_stopping1, early_stopping2, type_of_combination):
        self.early_stopping1 = early_stopping1
        self.early_stopping2 = early_stopping2
        self.type_of_combination = type_of_combination

    def __call__(self, history):
        if self.type_of_combination == "or":
            return self.early_stopping1(history) or self.early_stopping2(
                history)
        if self.type_of_combination == "and":
            return not (
                    (not self.early_stopping1(history))
                    or (not self.early_stopping2(history))
            )
