# -*- coding: utf-8 -*-
import sys,codecs
import numpy as np
import pandas as pd
import sys
filename=sys.argv[1]
trees=int(sys.argv[2])
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=crossv, test_size=0.09, random_state=54)
clf=RandomForestClassifier(criterion='entropy',n_estimators=trees, max_depth=None,min_samples_split=2, random_state=54,n_jobs=-1)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores,scores.mean(),scores.std())
