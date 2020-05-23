import numpy as np
import pandas as pd
from gui import *


class Naive_Bayes:
    def __init__(self):
        self.classes_name = []
        self.predicted = []
        self.zero_prob, self.one_prob = 0, 0
        self.zero_mean, self.one_mean = [], []
        self.zero_var, self.one_var = [], []
        self.zero_final, self.one_final = [], []
        self.training_dataset = pd.read_excel('parktraining.xlsx', header=None)
        self.testing_dataset = pd.read_excel('parktesting.xlsx', header=None)
        self.accuracy = 0

    # run naive bayes classification algorithm
    def run_naive_bayes(self):
        self.feature_scaling_train()
        self.class_probability()
        self.mean_var()
        self.feature_scaling_test()
        self.bayes_calculate()
        self.classifier()
        self.accuracy_calculate()

    # scale training features so that they have similar ranges
    def feature_scaling_train(self):
        col_count = len(self.training_dataset.columns)
        row_count = len(self.training_dataset.index)
        for i in range(col_count-1):
            col = self.training_dataset.loc[:, i]
            min_val = col.min()
            max_val = col.max()
            for j in range(row_count):
                val = (col[j] - min_val)/(max_val - min_val)
                self.training_dataset.at[j, i] = val

    # find probability of being zero/one
    def class_probability(self):
        col_count = len(self.training_dataset.columns)
        row_count = len(self.training_dataset.index)
        col = self.training_dataset.loc[:, col_count-1]
        num_zeros = 0
        num_ones = 0
        for i in range(row_count):
            if col[i] == 0:
                num_zeros += 1
            else:
                num_ones += 1

        self.zero_prob = num_zeros / row_count
        self.one_prob = num_ones / row_count

    # calculate mean/var of class zero/one
    def mean_var(self):
        col_count = len(self.training_dataset.columns)
        row_count = len(self.training_dataset.index)
        for i in range(col_count-1):
            temp0 = []
            temp1 = []
            col = self.training_dataset.loc[:, i]
            for j in range(row_count):
                if self.training_dataset.at[j, col_count-1] == 0:
                    temp0.append(col[j])
                else:
                    temp1.append(col[j])

            self.zero_mean.append(np.mean(temp0))
            self.one_mean.append(np.mean(temp1))
            self.zero_var.append(np.var(temp0))
            self.one_var.append(np.var(temp1))

    # scale testing features so that they have similar ranges
    def feature_scaling_test(self):
        col_count = len(self.testing_dataset.columns)
        row_count = len(self.testing_dataset.index)
        for i in range(col_count-1):
            col = self.testing_dataset.loc[:, i]
            min_val = col.min()
            max_val = col.max()
            for j in range(row_count):
                val = (col[j] - min_val)/(max_val - min_val)
                self.testing_dataset.at[j, i] = val

    # calculate final bayes values of zero/one
    def bayes_calculate(self):
        col_count = len(self.testing_dataset.columns)
        row_count = len(self.testing_dataset.index)
        for i in range(row_count):
            product_Zero = 1
            product_One = 1
            for j in range(col_count-1):

                pi0 = np.sqrt(2*np.pi*self.zero_var[j])
                sq0 = np.square(
                    self.testing_dataset.at[i, j] - self.zero_mean[j])
                tv0 = 2*self.zero_var[j]
                exp0 = -np.exp(-(sq0/tv0))
                ans0 = exp0/pi0
                product_Zero *= ans0

                pi1 = np.sqrt(2*np.pi*self.one_var[j])
                sq1 = np.square(
                    self.testing_dataset.at[i, j] - self.one_mean[j])
                tv1 = 2*self.one_var[j]
                exp1 = np.exp(-(sq1/tv1))
                ans1 = exp1/pi1
                product_One *= ans1

            product_Zero *= self.zero_prob
            product_One *= self.one_prob

            self.zero_final.append(product_Zero)
            self.one_final.append(product_One)

    # classify if zero/one
    def classifier(self):
        for i in range(len(self.one_final)):
            if self.zero_final[i] > self.one_final[i]:
                self.classes_name.append(0)
            else:
                self.classes_name.append(1)

    # calculate final accuracy of model
    def accuracy_calculate(self):
        col = self.testing_dataset.loc[:, len(self.testing_dataset.columns)-1]
        self.predicted = col
        count = 0
        for i in range(len(col)):
            if col[i] == self.classes_name[i]:
                count += 1

        frac = count/len(col)
        self.accuracy = frac * 100


if __name__ == "__main__":

    bayes = Naive_Bayes()
    bayes.run_naive_bayes()

    classes = ["Negative", "Positive"]
    for i in range(len(bayes.classes_name)):
        print(bayes.classes_name[i], bayes.predicted[i],
              bayes.classes_name[i] == bayes.predicted[i])
    print('Accuracy: %0.2f' % bayes.accuracy)

    tb = Table(class_names=classes, col1=bayes.classes_name,
               col2=bayes.predicted, value=bayes.accuracy)
    tb.create_Gui()
