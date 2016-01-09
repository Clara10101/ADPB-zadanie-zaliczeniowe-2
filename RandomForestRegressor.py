import numpy as np
import random
from collections import deque, Counter

inf = float('inf')

class BinNode:

    def __init__(self, values, left=None, right=None, condition=None, decision=None):

        self.left = left
        self.right = right
        self.values = values
        self.condition = condition
        self.decision = decision

    def is_leaf(self):

        if self.son("L") == None and self.son("R") == None:
            return True
        return False

    def son(self, which):

        if which == 'L':
            return self.left
        elif which == 'R':
            return self.right

    def set_sons_values(self, n_features, X, y):

        condition = self.find_best_division(n_features, X, y)


        if condition[0] != None and len(set(self.values_classes(y))) > 3:

            self.condition = condition[0]

            self.left = BinNode(condition[1])
            self.right = BinNode(condition[2])

            self.left.set_sons_values(n_features, X, y)

            self.right.set_sons_values(n_features, X, y)


        if len(set(self.values_classes(y))) <= 3:
            self.decision = np.average(self.values_classes(y))


    def values_classes(self, y):
       return y[self.values]


    def find_best_division(self, n_features, X, y):

        #wyznaczenie optymalnego podzialu w wezle
        #Dla kazdego wierzcholka bedziemy losowali n_features cech i tylko dla nich bedziemy sprawdzali wszystkie mozliwe wartosci
        #kryterium optymalnosci RSS
        #zwraca krotke postaci ((kolumna dla kryterium podzialu, kryterium podzialu, typ kolumny podzialu liczbowy/wyliczeniowy), obserwacje w lewym synu, obserwacje w prawym synu)
        #jesli nie ma mozliwosci ustanowienia takiego podzialu aby kazdy z synow zawieral jakies wartosci krotka postaci (None, None, None)

        m, n = X.shape
        data_type, classifier_classes = analyse_input_data(X, y)
        random_features = random.sample(range(0, n), n_features)


        best_division_condition = None; best_L_values = None; best_R_values = None;
        best_rss = inf

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
                    else: #wyliczeniowe
                        if X[value,i] == j:
                            L_values.append(value)
                        else:
                            R_values.append(value)
                n_L = len(L_values); n_R = len(R_values)

                #interesuje nas tylko podzial dzie kazdy z synow ma przypisane jakies obserwacje
                if n_L != 0 and n_R != 0:
                    rss = RSS({"L": L_values,"R":R_values})
                    #sprawdzenie czy do tej pory najlepszy warunek - minimalizuje gini
                    if rss < best_rss:
                        best_division_condition = (i,j,data_type[i][0])
                        best_L_values = L_values
                        best_R_values = R_values


        return best_division_condition, best_L_values, best_R_values

    def classify(self,v):
        #klasyfikuje wektor v na podstawie drzewa zawieszonego w wierzcholku self

        if not self.is_leaf():
            if self.condition[2] == 'numeryczne':
                if float(v[self.condition[0]]) <= float(self.condition[1]):
                    return self.son("L").classify(v)
                else:
                    return self.son("R").classify(v)
            else: #wyliczeniowe
                if v[self.condition[0]] == self.condition[1]:
                    return self.son("L").classify(v)
                else:
                    return self.son("R").classify(v)
        else:
            return self.decision

    def __repr__(self):

        if self.is_leaf():
            return 'Leaf(%r, %r)' % (self.values, self.decision)
        else:
            return 'Node(%r, %r)' % (self.values, self.condition)


class BinTree:

    def __init__(self, n_features, X, y):

        m, n = X.shape
        self.n_features = n_features
        self.X = X
        self.y = y

        self.node = BinNode([i for i in range(m)])
        print(self.node)
        self.node.set_sons_values(n_features, X, y)

    def root(self):
        return self.node

    def classify(self,v):
        return self.root().classify(v)



class RandomForestRegressor:

    def __init__(self, n_features):

        self.n_features = n_features
        self.forest = []
        self.classifier_classes = []

    def fit(self, X, y):

        m, n = X.shape

        new_tree = True
        last_oob_errors = deque(maxlen=11)

        #zakladamy ze tylko dwie klasy
        #jako pierwsza w liscie ma byc klasa y[0] - wystepujaca w zbiorze treneigowym jako pierwsza

        self.classifier_classes = [y[0]]
        for c in y:
            if c != y[0]:
                self.classifier_classes.append(c)

        #klasyfikacje dokonane za pomoca k-1 drzew
        actual_classification_for_training_set = [Counter({self.classifier_classes[0] : 0, self.classifier_classes[1] : 0}) for i in range(m)]

        while new_tree:

            #losowanie ze zwracaniem m przykladow ze zbioru treningowego
            rows = np.random.choice(m, m, replace=True)
            training_set = X[rows,:]
            training_set_classes = y[rows]

            #wiersze ktore nie sa w zbiorze treningowym - out of bag
            not_rows = list(set(range(m)).difference(rows))
            testing_set = X[not_rows,:]
            #testing_set_classes = y[not_rows]


            tree = self.create_decision_tree(training_set, training_set_classes)
            self.forest.append(tree)

            #dla kazdej obserwacji sprawdzamy decyzje utworzone przez dodane drzewo
            for i,row in enumerate(testing_set):
                decision = tree.classify(row)
                actual_classification_for_training_set[i][decision] += 1

            true_sum = 0
            false_sum = 0

            #zliczenie klasyfikacji po dodaniu drzewa
            for i,observation in enumerate(actual_classification_for_training_set):
                forest_decision = observation.most_common(1)[0][0]
                if forest_decision == y[i]:
                    true_sum += 1
                else:
                    false_sum += 1

            #aktualny blad oob po dodaniu do lasu drzewa
            oob_error = true_sum / float(true_sum + false_sum)
            last_oob_errors.append(oob_error)

            #sprawdzenie czy mozna zakonczyc proces uczenia nowych drzew
            if len(last_oob_errors) == 11:
                if list(last_oob_errors)[0] - (sum(list(last_oob_errors)[1:]) / 10.) < 0.01:
                    new_tree = False

    def predict(self,X):

        decisions = self.decisions_made_by_all_trees(X)
        forest_decisions = []
        print("decisions = ",decisions)

        for dec in decisions:
            forest_decisions.append(dec.most_common(1)[0][0])

        return forest_decisions

    def predict_proba(self,X):
        #zwraca wektor prawdopodobienstw przynaleznosci przykladow do pierwszej klasy - czyli do klasy wystepujacej w zbiorze treningowym jako pierwsza

        decisions = self.decisions_made_by_all_trees(X)
        forest_proba_decisions = []
        n_trees = float(len(self.forest))

        for dec in decisions:
            forest_proba_decisions.append(dec[self.classifier_classes[0]] / n_trees)

        return forest_proba_decisions

    def decisions_made_by_all_trees(self,X):

        m, n = X.shape

        #dla kazdego wiersza z tabeli X zliczenie decyzji podjetych przez wszystkie drzewa w lesie
        decisions = [Counter({self.classifier_classes[0] : 0, self.classifier_classes[1] : 0}) for i in range(m)]
        for i,row in enumerate(X):
            for tree in self.forest:
                decision = tree.classify(row)
                decisions[i][decision] += 1
        return decisions

    def create_decision_tree(self, X, y):

        tree = BinTree(self.n_features, X, y)
        return tree



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


def RSS(data):

        y1, y2 = [np.average(data[key]) for key in data]
        s1 = sum([(y1-yi)**2 for yi in list(data.values())[0]])
        s2 = sum([(y1-yi)**2 for yi in list(data.values())[1]])
        return s1+s2

def showR(node, prefix=''):
    if node.is_leaf():
        return prefix + '-' + str(node) + '\n'
    else:
        return showR(node.son('L'),prefix+'   ') + prefix + '-<' + '\n' + showR(node.son('R'),prefix+'   ')

def show(tree):
    return showR(tree.root())



# Dane testowe
dane_test_X = np.array(
    [['Honda', float(2009), 'igla',10]
    , ['Honda', float(2005), 'igla',9],
     ['Honda', float(2006), 'idealny',9],
     ['Renault', float(2010), 'igla',10],
     ['Renault', float(2007), 'idealny',9],
     ['Ford', float(2009), 'igla',7]
    , ['Ford', float(2005), 'igla',6],
     ['Fiat', float(2006), 'idealny',6 ],
     ['Mazda', float(2010), 'igla',7,],
     ['Opel',float(2007), 'idealny',8]
     ])
dane_test_y = np.array([180000, 10100, 215000,130000, 200000,
                        13000,15000,21000,80000,50000
                        ])

#analyse_input_data(dane_test_X, dane_test_y)
#przykladowy wynik wywolania
#[('wyliczeniowe', ['Honda']), ('numeryczne', ['2009', '2006', '2005']), ('wyliczeniowe', ['igla', 'idealny']), ('numeryczne', ['180000.87', '10100', '215000'])]

r = RandomForestRegressor(2)
tree = r.create_decision_tree(dane_test_X, dane_test_y)
print (tree.root())
print (show(tree))

przyklad_testowy = ['Renault', 2005, 'bezkolizyjny', 215000]
#print( tree.classify(przyklad_testowy))

r.fit(dane_test_X,dane_test_y)
print (r.forest)
print (r.predict(dane_test_X))
#print (r.predict_proba(dane_test_X))
