import numpy as np
import random

class BinNode:
    def __init__(self, values, left=None, right=None):
        self.left = left
        self.right = right
        self.values = values

    def is_leaf(self):
        if self.son("L") == None and self.son("R") == None:
            return True
        return False

    def son(self, which):

        if which == 'L':
            return self.left
        elif which == 'R':
            return self.right

    def find_best_division(self, n_features, X, y):
        #wyznaczenie optymalnego podzialu w wezle
        #Dla kazdego wierzcholka bedziemy losowali n_features cech i tylko dla nich będziemy sprawdzali wszystkie mozliwe wartosci,
        #kryterium optymalnosci jest Gini impurity
        m, n = X.shape
        data_type = analyse_input_data(X, y)
        random_features = random.sample(range(1, n), n_features)
        for i in random_features:
            #sprawdzenie wszystkich mozliwych wartosci dla kazdej z wylosowanych cech
            if data_type[0] == 'numeryczne':
                #sprawdzenie kazdej wartosci wezlowej
                pass


    def __repr__(self):
        return 'Tree(%r, %r, %r)' % (self.values, self.left, self.right)


class BinTree:
    def __init__(self, node):
        self.node = node

    def root(self):
        return self.node


class RandomForestClassifier:
    def __init__(self, n_features):
        self.n = n_features

    def fit(self, X, y):
        pass

    def create_decision_tree(self, X, y):
        m, n = X.shape

        tree = BinTree([i for i in range(m)])
        # znalezienie najlepszego warunku podzialu


def analyse_input_data(X, y):
    """

    :param X: dane treningowe, tablica numpy array wymiaru (m x n)
    :param y: klasy dla zbioru treningowego, numpy array dlugosci m
    :return:
    """
    m, n = X.shape
    data_type = [() for i in range(n)]
    for i in range(n):
        # sprawdzenie typu danych dla kazdej kolumny
        wartosci = []
        if all(map(lambda x: x.replace('.', '', 1).isdigit(), X[:, i])):
            typ = "numeryczne"
        else:
            typ = "wyliczeniowe"
        wartosci.extend(list(set(X[:, i])))
        data_type[i] = (typ, wartosci)
    print data_type
    return data_type


def is_numeric(v):
    for i in range(len(v)):
        # if v[i]
        pass


'''
Trzeba wybrac ceche ktora najlepiej podzieli zbior.
Losujemy n_features i z nich wybieramy za pomoca kryterium Gini impurity
'''

'''
gini = kryterium optymalnosci Gini impurity.
n - liczba wszystkich przykladow
nl, nr - liczba przykladow, ktore po podziale trafia do lewego i prawego syna
nl0, nl1 - liczba przykladow z pierwszej i drugiej klasy w lewym synu
np0, np1 - analogicznie jakw wyzej tylko dla prawego syna
'''


def gini(n, nl, nr, nl0, nl1, nr0, nr1):
    return (nl / n) * (nl0 / nl * (1 - nl0 / nl) + nl1 / nl * (1 - nl1 / nl)) + (nr / n) * (
    nr0 / nr * (1 - nr0 / nr) + nr1 / nr * (1 - nr1 / nr))


# t=Tree(5,Tree(6,7),7)
#print t
dane_test_X = np.array(
    [['Honda', 2009, 'igla', 180000.87], ['Honda', 2005, 'igla', 10100], ['Honda', 2006, 'idealny', 215000]])
dane_test_y = np.array(['KUP', 'NIE_KUPUJ', 'NIE_KUPUJ'])
analyse_input_data(dane_test_X, dane_test_y)