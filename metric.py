import numpy as np
from numpy import genfromtxt

def get_data():
    training_samples = genfromtxt('../data/training_set.csv', delimiter=',', usecols = range(41))
    training_labels = genfromtxt('../data/training_labels.txt', delimiter=',', usecols = range(1) , dtype=None)
    validation_samples = genfromtxt('../data/validation_set.csv', delimiter=',', usecols = range(41))
    validation_labels = genfromtxt('../data/validation_labels.txt', delimiter=',', usecols = range(1) , dtype=None)
    test_samples = genfromtxt('../data/test_set.csv', delimiter=',', usecols = range(41))
    return training_samples, validation_samples, test_samples, training_labels, validation_labels

