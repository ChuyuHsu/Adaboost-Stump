#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from decisionstump import DecisionStump
from adaboost import AdaBoost


def main():
    data = np.loadtxt(open("/Users/rio512hsu/dataset/MachineLearningTechniques" +
                           "/hw2_adaboost_train.csv", "rb"),
                      delimiter=" ")

    X = data[:, :-1]
    y = data[:, -1]
    u = np.ones(X.shape[0]) / X.shape[0]
    clf = DecisionStump().fit(X, y, u)
    # Q12
    print clf.getEin()

    # Q13
    adaboost = AdaBoost(DecisionStump).fit(X, y, 300)
    # print adaboost.predict(X)
    print np.sum(adaboost.predict(X) != y)

    # Q17
    test = np.loadtxt(open("/Users/rio512hsu/dataset/" +
                           "MachineLearningTechniques/" +
                           "hw2_adaboost_test.csv"),
                      delimiter=' ')
    X_test = test[:, :-1]
    y_test = test[:, -1]
    print np.sum(clf.predict(X) != y) / float(test.shape[0])

    # Q18
    print np.sum(adaboost.predict(X_test) != y_test) / float(test.shape[0])

    return 0


if __name__ == "__main__":
    main()
