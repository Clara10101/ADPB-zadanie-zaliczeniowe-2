# ADPB-zadanie-zaliczeniowe-2
## Moduł do uczenia pod nadzorem lasów losowych, które mogą być użyte do klasyfikacji i regresji.

Moduł udostępnia dwie klasy z następującymi metodami:

- RandomForestClassifier
  - fit(X, y) - uczy klasyfikator na zbiorze treningowym X; y jest wektorem, który dla każdego wiersza X zawiera klasę, do której należy dany przykład
  - predict(X) - zwraca wektor najbardziej prawdopodobnych klasy dla obserwacji w X
  - predict_proba(X) - zwraca wektor prawdopodobieństw przynależności przykładów z X do klasy występującej w zbiorze treningowym jako pierwsza

- RandomForestRegressor
  - fit(X, y) - uczy klasyfikator na zbiorze treningowym X; y jest wektorem,który dla każdego wiersza X zawiera wartość zmiennej zależnej
  - predict(X) - zwraca wektor wyników regresji dla przykładów w X

X jest tablicą dwuwymiarową o wymiarach (m x n), a y wektorem długości m. Zarówno X jak i y powinny być typu numpy.array.
Konstruktory klas RandomForestClassifier oraz RandomForestRegressor przyjmują jeden parametr w postaci liczby naturalnej >= 1.

------------------------------------------

Poniżej można znaleźć kilka przykładów wykorzystania modułu:

1. Klasyfikacja dla zbioru danych na temat sprzedarzy samochodów

```
import RandomForestClassifier, RandomForestRegressor
import numpy as np

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

cars_training_y = np.array(
    ['KUP', 'NIE_KUPUJ', 'NIE_KUPUJ','KUP', 'NIE_KUPUJ', 'NIE_KUPUJ', 'KUP', 'KUP'])

cars_training_z = np.array(
    [180000, 10100, 215000,130000, 200000, 13000,15000,21000])

cars_testing_X = np.array(
    [['Ford', 2007, 'idealny', 230000],
     ['Fiat', 2014, 'bezkolizyjny', 198000],
     ['Renault', 2000, 'igla', 1560000]
    ])

#Klasyfikacja
c = RandomForestClassifier.RandomForestClassifier(3)
c.fit(cars_training_X,cars_training_y)
c.predict(cars_testing_X)
c.predict_proba(cars_testing_X)

#Regresja
r = RandomForestRegressor.RandomForestRegressor(3)
r.fit(cars_training_X, cars_training_z)
r.predict(cars_training_X)
```







