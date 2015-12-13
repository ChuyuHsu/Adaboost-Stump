#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math


class DecisionStump(object):
    """docstring for DecisionStump
    """
    def __init__(self):
        self.__theta = None
        self.__sign = None
        self.__Ein = None
        self.__width = None # number of features in a data instance
        self.__i = None # index of feature

    def predict(self, X):
        return self.__predict(X, self.__i, self.__theta, self.__sign)

    def __predict(self, X, i, theta, sign):
        if np.shape(X) != self.__width:
            raise Error("Features of data doesn't fit that of the training data")

        y = np.zeros(np.shape(X)[0])

        if sign == 1:
            y = np.where(X[:, i] > theta, 1, -1)
        else:
            y = np.where(X[:, i] < theta, 1, -1)

        return y


    def fit(self, X, y, u):
        m, n = np.shape(X)
        self.__width = n

        stumps = [] # list of (sign, theta, Ein)
        sorted_index = [np.argsort(X[:, i]) for i in xrange(n)]

        for i in range(n):
            xi = X[sorted_index[i],i]
            yi = y[sorted_index[i]]
            ui = u[sorted_index[i]]
            stumps.append((i,) + self.__getStump(xi, yi, ui))

        best = min(stumps, key=lambda s: s[3])
        # (i of feature, sign, theta, Ein)
        self.__i, self.__sign, self.__theta, self.__Ein = best
        return self


    def __getStump(self, x, y, u):
        """
        output: best stump in this feature xi
        s, theta, Ein
        """
        thetas = np.array([float("-inf")] +
                          [(x[i] + x[i + 1])/2
                          for i in range(0, x.shape[0] - 1)] +
                          [float("inf")])

        Ein = sum(u)
        sign = 1
        target_theta = 0.0
        # positive and negative rays
        for theta in thetas:
            y_positive = np.where(x > theta, 1, -1)
            y_negative = np.where(x < theta, 1, -1)

            weighted_error_positive = sum((y_positive != y)*u)
            weighted_error_negative = sum((y_negative != y)*u)
            # consider sign s, choose min Ein
            if weighted_error_positive > weighted_error_negative:
                if Ein > weighted_error_negative:
                    Ein = weighted_error_negative
                    sign = -1
                    target_theta = theta
            else:
                if Ein > weighted_error_positive:
                    Ein = weighted_error_positive
                    sign = 1
                    target_theta = theta

        if target_theta == float('inf'):
            target_theta = 1.0
        elif target_theta == float('-inf'):
            target_theta = -1.0

        return (sign, theta, Ein)

    def getEin():
        return self.__Ein
