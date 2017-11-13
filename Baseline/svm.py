#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 03:20:41 2017

@author: csuser
"""

import pandas as pd
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

fileName = '../data/kddcup.data_10_percent';
#my_data = genfromtxt(fileName, delimiter=',')
#print my_data;

df=pd.read_csv(fileName, sep=',',header=None)
print df.values;
df = df.values;
print df.shape;
print df[0][41];
print df[0][4];


dict_1 = {};
dict_2 = {};
dict_3 = {};

numValues_1 = 0;
for i in range(0,df.shape[0]):
    value = df[i][1];
    if(value not in dict_1):
        numValues_1 = numValues_1 + 1;
        dict_1[value] = numValues_1;
print dict_1;
print len(dict_1);
        
numValues_2 = 0;
for i in range(0,df.shape[0]):
    value = df[i][2];
    if(value not in dict_2):
        numValues_2 = numValues_2 + 1;
        dict_2[value] = numValues_2;
print dict_2;
print len(dict_2);

numValues_3 = 0;
for i in range(0,df.shape[0]):
    value = df[i][3];
    if(value not in dict_3):
        numValues_3 = numValues_3 + 1;
        dict_3[value] = numValues_3;
print dict_3;
print len(dict_3);

N = df.shape[0];
p = df.shape[1]-1
X = np.zeros((N,p), dtype = float);
print X.shape
X = df[:,:p];
print X
print X.shape

for i in range(0, X.shape[0]):
    value = X[i][1];
    if(value in dict_1):
        X[i][1] = dict_1[value];
    else:
        print 'error: dict_1'
print X
print X.shape

for i in range(0, X.shape[0]):
    value = X[i][2];
    if(value in dict_2):
        X[i][2] = dict_2[value];
    else:
        print 'error: dict_2'
print X
print X.shape

for i in range(0, X.shape[0]):
    value = X[i][3];
    if(value in dict_3):
        X[i][3] = dict_3[value];
    else:
        print 'error: dict_3'
print X
print X.shape

np.random.shuffle(X);
print X
print X.shape

labels = np.zeros((1,p+1), dtype = float);
labels = df[:,p];
print labels
print labels.shape

for i in range(0, len(labels)):
    l = len(labels[i]);
    temp = labels[i]
    labels[i] = temp[0:l-1]
print labels
print labels.shape    

trainingDataSize =  int(math.ceil(0.7*N));
validationDataSize = int(math.ceil(0.1*N));
testDataSize = N - trainingDataSize - validationDataSize;

print trainingDataSize
print validationDataSize
print testDataSize

trainingData = X[0:trainingDataSize,:]
validationData = X[trainingDataSize+1:trainingDataSize+validationDataSize,:]
testData = X[trainingDataSize+validationDataSize+1:trainingDataSize+validationDataSize+testDataSize,:]

trainingLabels = labels[0:trainingDataSize]
validationLabels = labels[trainingDataSize+1:trainingDataSize+validationDataSize]
testLabels = labels[trainingDataSize+validationDataSize+1:trainingDataSize+validationDataSize+testDataSize]

print trainingData
print trainingDataSize
print validationData
print validationDataSize
print testData
print testDataSize

clf = SVC()
clf.fit(trainingData, trainingLabels);

y_pred = clf.predict(testData);

print precision_recall_fscore_support(testLabels, y_pred)
