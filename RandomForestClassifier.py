import numpy as np
import random

inf = float('inf')

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
        #Dla kazdego wierzcholka bedziemy losowali n_features cech i tylko dla nich bedziemy sprawdzali wszystkie mozliwe wartosci
        #kryterium optymalnosci Gini impurity
        #zwraca krotke postaci (kryterium podzialu, obserwacje w lewym synu, obserwacje w prawym synu)
        #jesli nie ma mozliwosci ustanowienia takiego podzialu aby kazdy z synow zawieral jakies wartosci krotka postaci (None, None, None)

        m, n = X.shape
        data_type, classifier_classes = analyse_input_data(X, y)
        random_features = random.sample(range(1, n), n_features)
        best_division_condition = None; best_L_values = None; best_R_values = None
        best_gini_impurity = inf

        #sprawdzenie wszystkich mozliwych wartosci dla kazdej z wylosowanych cech
        for i in random_features:

            #sprawdzenie kazdej wartosci wezlowej
            for j in data_type[i][1]:
                L_values = []; R_values = []
                for value in self.values:
                    if data_type[i][0] == 'numeryczne':
                        if float(X[value,i]) <= float(j):
                            L_values.append(value)
                        else:
                            R_values.append(value)
                    else:
                        if X[value,i] == j:
                            L_values.append(value)
                        else:
                            R_values.append(value)

                n_L = len(L_values); n_R = len(R_values)
                n_L0 = sum([val == classifier_classes[0] for val in y[L_values]])
                n_L1 = sum([val == classifier_classes[1] for val in y[L_values]])#n-n_l0
                n_R0 = sum([val == classifier_classes[0] for val in y[R_values]])
                n_R1 = sum([val == classifier_classes[1] for val in y[R_values]])

                #interesuje nas tylko podzial dzie kazdy z synow ma przypisane jakies obserwacje
                if n_L != 0 and n_R != 0:
                    gini_impurity = gini(len(self.values),n_L,n_R,n_L0,n_L1,n_R0,n_R1)

                    #sprawdzenie czy do tej pory najlepszy warunek - minimalizuje gini
                    if gini_impurity < best_gini_impurity:
                        best_gini_impurity = gini_impurity
                        best_division_condition = j
                        best_L_values = L_values
                        best_R_values = R_values

        return best_division_condition, best_L_values, best_R_values



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
    Sprawdza typ danych wejsciowych, jakiego typu sa cechy, czy numeryczne, cz wyliczeniowe i jakie wartosci przyjmuja
    :param X: dane treningowe, tablica numpy array wymiaru (m x n)
    :param y: klasy dla zbioru treningowego, numpy array dlugosci m
    :return:
    """
    m, n = X.shape
    data_type = [() for i in range(n)]
    for i in range(n):
        # sprawdzenie typu danych dla kazdej kolumny
        wartosci = []
        if all(map(is_numeric , X[:, i])):
            typ = "numeryczne"
        else:
            typ = "wyliczeniowe"
        wartosci.extend(list(set(X[:, i])))
        data_type[i] = (typ, wartosci)
    #print data_type
    classifier_classes = list(set(y))
    return data_type, classifier_classes


def is_numeric(x):
    return x.replace('.', '', 1).isdigit()


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
    [['Honda', 2009, 'igla', 180000.87], ['Honda', 2005, 'igla', 10100], ['Honda', 2006, 'idealny', 215000], ['Renault', 2010, 'igla', 130000], ['Renault', 2007, 'idealny', 200000]])
dane_test_y = np.array(['KUP', 'NIE_KUPUJ', 'NIE_KUPUJ','KUP', 'NIE_KUPUJ'])
#analyse_input_data(dane_test_X, dane_test_y)
#przykladowy wynik wywolania
#[('wyliczeniowe', ['Honda']), ('numeryczne', ['2009', '2006', '2005']), ('wyliczeniowe', ['igla', 'idealny']), ('numeryczne', ['180000.87', '10100', '215000'])]
wierzchilek1 = BinNode([0,1,2,3,4])
print wierzchilek1.find_best_division(3,dane_test_X, dane_test_y)