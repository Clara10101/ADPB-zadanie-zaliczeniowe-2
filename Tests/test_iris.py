__author__ = 'Klara'

from RandomForest import RandomForestRegressor, RandomForestClassifier
import numpy as np
from sklearn import datasets

#Test dla danych o samochodach
cars_training_X = np.array(
    [['Honda', 2009, 'igla', 180000.87],
     ['Honda', 2005, 'igla', 10100],
     ['Honda', 2006, 'idealny', 215000],
     ['Renault', 2010, 'igla', 130000],
     ['Renault', 2007, 'idealny', 200000],
     ['Renault', 2005, 'bezkolizyjny', 215000],
     ['Ford', 2008, 'bezkolizyjny',225000],
     ['Fiat', 2012, 'igla', 130000]
    ])

cars_training_y = np.array(['KUP', 'NIE_KUPUJ', 'NIE_KUPUJ','KUP', 'NIE_KUPUJ', 'NIE_KUPUJ', 'KUP', 'KUP'])

cars_training_z = np.array([180000, 10100, 215000,130000, 200000, 13000,15000,21000])

cars_testing_X = np.array(
    [['Ford', 2007, 'idealny', 230000],
     ['Fiat', 2014, 'bezkolizyjny', 198000],
     ['Renault', 2000, 'igla', 1560000]
    ])

#Cars test dla klasyfikacji
"""r1 = RandomForestClassifier.RandomForestClassifier(3)
r1.fit(cars_training_X,cars_training_y)
print r1.predict(cars_testing_X)
print r1.predict_proba(cars_testing_X)"""

#Cars test dla regresji
r = RandomForestRegressor.RandomForestRegressor(3)
r.fit(cars_training_X, cars_training_z)
print "Test regresja"
print r.predict(cars_training_X)

#Test dla danych iris
iris = datasets.load_iris()
iris_y = np.array(iris.target[:100], dtype=str)
iris_X = np.array(iris.data[:100])

rows = np.random.choice(100, 80, replace=False)
iris_training_X = iris_X[rows,:]
iris_training_y = iris_y[rows]

not_rows = list(set(range(100)).difference(rows))
iris_testing_X = iris_X[not_rows,:]
iris_testing_y = iris_y[not_rows]

#inicjalizacja klasyfikatora
r2 = RandomForestClassifier.RandomForestClassifier(3)

#nauczenie klasyfikatora na zbiorze treningowym
r2.fit(iris_training_X,iris_training_y)

#predukcje
predykcje_r2 = r2.predict(iris_testing_X)
print predykcje_r2

print r2.predict_proba(iris_testing_X)
print iris_testing_y

print "Poprawnie sklasyfikowanych " + str(sum(predykcje_r2 == iris_testing_y)) + "/20"