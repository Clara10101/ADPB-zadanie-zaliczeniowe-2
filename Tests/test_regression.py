## regresja

import numpy
import random
from sklearn.ensemble import RandomForestRegressor as sklearnRandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from RandomForest import RandomForestRegressor

f = open("forestfires_log.csv")
X = []
y = []
for l in f:
    l=l.split()
    X.append(l[:-1])
    y.append(l[-1])
z = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
for i in range(len(X)):
    for v in z:
        X[i][v] = float(X[i][v])
for i in range(len(y)):
    y[i] = float(y[i])
le = preprocessing.LabelEncoder()
months = map(lambda x: x[2], X)
le.fit(months)
months_labels = le.transform(months)
days = map(lambda x: x[3], X)
le.fit(days)
days_labels = le.transform(days)
for i in range(len(X)):
    X[i][2] = months_labels[i]
    X[i][3] = days_labels[i]

paired = zip(X, y)
random.shuffle(paired)
X, y = zip(*paired)
X = numpy.array(X)
y = numpy.array(y)

m, n = X.shape

"""def cross_val_scores(r, X, y, cv):
    kf = cross_validation.KFold(n=len(X), n_folds=10)
    scores = []
    for train, test in kf:
        r.fit(X[train], y[train])
        scores.append(mean_squared_error(r.predict(X[test]), y[test]))
    return scores"""

treningowe = numpy.random.choice(m, m, replace=True)
testowe = list(set(range(m)).difference(treningowe))

r = RandomForestRegressor.RandomForestRegressor(10)
sr = sklearnRandomForestRegressor()
r.fit(X[treningowe],y[treningowe])
sr.fit(X[treningowe],y[treningowe])
print r.predict(X[testowe])
print sr.predict(X[testowe])
#scores = cross_val_scores(r, X, y, 10)
#sklearn_scores = cross_val_scores(sr, X, y, 10)
