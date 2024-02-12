# -*- coding: utf-8 -*-
import sys,codecs
import numpy as np
import pandas as pd
import sys
filename='24_30rand.csv'
trees=493
taco=pd.read_csv(filename)
taco.fillna(0,inplace=True)
crossv=10
print('filename:',filename)
print("data instances and parameters:",taco.shape)
print("cross-validation:",crossv)
print("trees:",trees)
y = taco['Normal']
X=taco.drop(['Normal','Event Id'],axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import *
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=crossv, test_size=0.09, random_state=54)
clf=RandomForestClassifier(criterion='entropy',n_estimators=trees, max_depth=None,min_samples_split=2, random_state=54,n_jobs=-1)
scoring={'accuracy' : make_scorer(accuracy_score),
        'precision' : make_scorer(precision_score),
        'recall' : make_scorer(recall_score),
        'f1_score' : make_scorer(f1_score)}
scores = cross_validate(clf, X, y, cv=cv,scoring=scoring)
print('test_accuracy:',scores['test_accuracy'])
print('test_accuracy mean:',sum(scores['test_accuracy'])/10.)
print('test_precision:',scores['test_precision'])
print('test_precision mean:',sum(scores['test_precision'])/10.)
print('test_recall:',scores['test_recall'])
print('test_recall mean:',sum(scores['test_recall'])/10.)
print('test_f1_score:',scores['test_f1_score'])
print('test_f1_score mean:',sum(scores['test_f1_score'])/10.)
