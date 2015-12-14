#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from decisionstump import DecisionStump
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

        for t in xrange(iteration):
            clf = DecisionStump().fit(X, y, u)

            errorRate = clf.getEin() / sum(u)
            prediction = clf.predict(X)
            scalingFactor = math.sqrt(((1.0 - errorRate) / errorRate))
            print "sum(u): %f, Ein: %f, Erate: %f, SF: %f" %\
                (sum(u), clf.getEin(), errorRate, scalingFactor)
            u = (prediction != y) * u * scalingFactor +\
                (prediction == y) * u / scalingFactor
            # print "u: %s" % str(u)

            self.alphas.append(math.log(scalingFactor))
            self.gs.append(clf)
        # alpha = math.log(scalingFactor, math.e)

        return self

    def predict(self, X):
        ys = np.array([g.predict(X) for g in self.gs])
        # print "ys: %s" % str(ys)
        y = np.where(ys.T.dot(self.alphas) > 0, 1, -1)
        # print "y: %s" % str(y)
        return y
