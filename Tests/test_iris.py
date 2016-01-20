from ..RandomForest import RandomForestClassifier
import numpy as np
from sklearn import datasets

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
r = RandomForestClassifier.RandomForestClassifier(3)

#nauczenie klasyfikatora na zbiorze treningowym
r.fit(iris_training_X,iris_training_y)

#predukcje
predykcje_r = r.predict(iris_testing_X)
print predykcje_r

print r.predict_proba(iris_testing_X)
print iris_testing_y

print "Poprawnie sklasyfikowanych " + str(sum(predykcje_r == iris_testing_y)) + "/20"
