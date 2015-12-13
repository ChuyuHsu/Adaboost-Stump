#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from DecisionStump import DecisionStump
import math


class AdaBoost(object):

    """Docstring for AdaBoost. """

    def __init__(self, classifier):
        """TODO: to be defined1. """
        self.___classifier = classifier
        self.alphas = []
        self.gs = []

    def fit(self, X, y, iteration):
        u = np.ones(X.shape[0]) / X.shape[0]

        for t in iteration:
            clf = DecisionStump().fit(X, y, u)

            errorRate = clf.getEin() / sum(u)
            prediction = clf.predict(X)
            scalingFactor = math.sqrt((1 - errorRate / errorRate))
            u = scalingFactor * (prediction != y) * u +\
                (prediction == y) * u / scalingFactor

            self.alphas.append(math.log(scalingFactor))
            self.gs.append(clf)
        # alpha = math.log(scalingFactor, math.e)

    def predict(self, X):
        ys = np.array([g.predict(X) for g in self.gs])
        y = np.where(np.dot(ys, self.alphas) > 0, 1, -1)
        return y
