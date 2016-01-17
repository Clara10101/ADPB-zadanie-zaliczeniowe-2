import numpy as np
import random

inf = float('inf')

class BinNode:
    """
    Klasa reprezentujaca wezel w drzewie binarnym.
    """

    def __init__(self, values, data_type, classifier_classes=None, left=None, right=None, condition=None, decision=None):
        """
        Inicjalizacja wszystkich potrzebnych parametrow. Podanie jedynie wyktora values jest niezbedne.
        """

        self.left = left
        self.right = right
        self.values = values #obserwacje w danym wezle (ich numery wiersza w zbiorze treningowym X)
        self.condition = condition #warunek podzialu w wezle
        self.decision = decision #jesli wierzcholek jest lisciem to przypisywana jest mu decyzja klasyfikacji
        self.training_data_type = data_type
        self.classifier_classes = classifier_classes

    def is_leaf(self):
        """
        Sprawdzenie czy wezel jest lisciem
        """

        if self.son("L") == None and self.son("R") == None:
            return True
        return False

    def son(self, which):
        """
        Zwraca wybranego syna wezla.
        :param which: 'L' lub 'R'
        """

        if which == 'L':
            return self.left
        elif which == 'R':
            return self.right

    def set_sons_values(self, n_features, X, y, which='C'):
        """
        Przypisanie rekurencyjnie obserwacji do synow korzenia na podstawie wybranego najlepszego kryterium podzialu
        :param n_features: liczba calkowita wieksza od 1 i mniejsza od liczby wszystkich obserwacji w danych treningowych
        :param X: dane treningowe, tablica numpy array, w ktorej kazdy wiersz odpowiada jednej obserwacji
        :param y: wektor klasyfikacji dla danych treningowych, numpy array
        """

        condition = self.find_best_division(n_features, X, y, which)

        if which == 'C': #Klasyfikacja

            #jesli podzial jest mozliwy i potrzebny
            if condition[0] != None and len(set(self.values_classes(y))) > 1:

                self.condition = condition[0]
                self.left = BinNode(condition[1], self.training_data_type, self.classifier_classes)
                self.right = BinNode(condition[2], self.training_data_type, self.classifier_classes)

                self.left.set_sons_values(n_features, X, y, which='C')
                self.right.set_sons_values(n_features, X, y, which='C')

            if len(set(self.values_classes(y))) == 1:
                self.decision = list(set(self.values_classes(y)))[0]

        if which == 'R': #Regresja

            if condition[0] != None and len(self.values) > 3:

                self.condition = condition[0]

                self.left = BinNode(condition[1], self.training_data_type, self.classifier_classes)
                self.right = BinNode(condition[2],self.training_data_type, self.classifier_classes)

                self.left.set_sons_values(n_features, X, y, which='R')
                self.right.set_sons_values(n_features, X, y, which='R')

            else:

                self.decision = np.average(self.values_classes(y))


    def values_classes(self, y):
        return y[self.values]

    def find_best_division(self, n_features, X, y, which):
        """
        Wyznacza optymalny podzial w wezle.
        Dla kazdego wierzcholka losowanych jest n_features cech i dla nich sprawdzane sa wszystkie mozliwe wartosci kryterium optymalnosci Gini impurity.
        Zwraca krotke postaci ((kolumna dla kryterium podzialu, kryterium podzialu, typ kolumny podzialu liczbowy/wyliczeniowy), obserwacje w lewym synu, obserwacje w prawym synu)
        oraz wynik alalizy danych wejsciowych,
        jesli nie ma mozliwosci ustanowienia takiego podzialu aby kazdy z synow zawieral jakies wartosci zwracana jest krotka postaci (None, None, None)
        """

        m, n = X.shape
        data_type = self.training_data_type
        classifier_classes = self.classifier_classes

        #data_type, classifier_classes = analyse_input_data(X, y)

        random_features = random.sample(range(0, n-1), n_features)

        best_division_condition = None; best_L_values = None; best_R_values = None;

        #Najlepszy gini_impunity lub rss w zaleznosci od tego ktory rodzaj klasyfikacji
        best_gini_impurity = inf
        best_rss = inf

        n_all = len(self.values)

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

                if which == 'C': #Klasyfikacja

                    n_L0 = sum([val == classifier_classes[0] for val in y[L_values]])
                    n_L1 = n_all - n_L0
                    n_R0 = sum([val == classifier_classes[0] for val in y[R_values]])
                    n_R1 = n_all - n_R0

                    #interesuje nas tylko podzial gdzie kazdy z synow ma przypisane jakies obserwacje
                    if n_L != 0 and n_R != 0:
                        gini_impurity = gini(n_all,n_L,n_R,n_L0,n_L1,n_R0,n_R1)

                        #sprawdzenie czy do tej pory najlepszy warunek - minimalizuje gini
                        if gini_impurity < best_gini_impurity:
                            best_gini_impurity = gini_impurity
                            best_division_condition = (i,j,data_type[i][0])
                            best_L_values = L_values
                            best_R_values = R_values

                if which == 'R': #Regresja

                    #interesuje nas tylko podzial gdzie kazdy z synow ma przypisane jakies obserwacje
                    if n_L != 0 and n_R != 0:
                        rss = RSS({"L": L_values,"R":R_values})
                        #sprawdzenie czy do tej pory najlepszy warunek - minimalizuje rss
                        if rss < best_rss:
                            best_division_condition = (i,j,data_type[i][0])
                            best_L_values = L_values
                            best_R_values = R_values


        return best_division_condition, best_L_values, best_R_values

    def classify(self,v):
        """
        Klasyfikuje wektor v na podstawie drzewa zawieszonego w wierzcholku self.
        :param v: wektor postaci numpy array
        """

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
    """
    Klasa reprezentujaca binarne drzewo decyzyjne.
    """

    def __init__(self, n_features, X, y, data_type, classifier_classes, which='C'):

        m, n = X.shape
        self.n_features = n_features
        self.X = X
        self.y = y

        #typ danych na ktorych uczony byl klasyfikator tworzacy dane drzewo
        self.training_data_type = data_type
        self.classifier_classes = classifier_classes

        self.node = BinNode([i for i in range(m)],self.training_data_type, self.classifier_classes)
        self.node.set_sons_values(n_features, X, y, which)

    def root(self):
        return self.node

    def classify(self,v):
        return self.root().classify(v)

def is_numeric(x):
    return str(x).replace('.', '', 1).isdigit()

def gini(n, nl, nr, nl0, nl1, nr0, nr1):
    """
    Wyznacza kryterium optymalnosci Gini impurity
    :param n: liczba wszystkich przykladow
    :param nl: liczba przykladow, ktore po podziale trafia do lewego syna
    :param nr: liczba przykladow, ktore po podziale trafia do prawego syna
    :param nl0: liczba przykladow z pierwszej klasy w lewym synu
    :param nl1: liczba przykladow z drugiej klasy w lewym synu
    :param nr0: liczba przykladow z pierwszej klasy w prawym synu
    :param nr1: liczba przykladow z drugiej klasy w prawym synu
    :return: wartosc gini impurity
    """
    return (nl / n) * (nl0 / nl * (1 - nl0 / nl) + nl1 / nl * (1 - nl1 / nl)) + (nr / n) * (
    nr0 / nr * (1 - nr0 / nr) + nr1 / nr * (1 - nr1 / nr))


def RSS(data):

        y1, y2 = [np.average(data[key]) for key in data]
        s1 = sum([(y1-yi)**2 for yi in list(data.values())[0]])
        s2 = sum([(y1-yi)**2 for yi in list(data.values())[1]])

        return s1+s2

def analyse_input_data(X, y=[]):
    """
    Sprawdza typ danych wejsciowych, jakiego typu sa cechy, czy numeryczne, cz wyliczeniowe i jakie wartosci przyjmuja
    :param X: dane treningowe, tablica numpy array wymiaru (m x n)
    :param y: klasy dla zbioru treningowego, numpy array dlugosci m
    :return: krotka postaci ([(typ_kolumna_0,wartosci_kolumna_0), ... , (typ_kolumna_n,wartosci_kolumna_n)],[klasa1,klasa2])
    """

    m, n = X.shape

    data_type = [() for i in range(n)]

    if len(y) != 0:
        #sprawdzenie pierwszego wymiaru X i dlugosci y
        if not m == len(y):
            raise ValueError

    for i in range(n):
        # sprawdzenie typu danych dla kazdej kolumny
        wartosci = []
        if all(map(is_numeric , X[:, i])):
            typ = "numeryczne"
        else:
            typ = "wyliczeniowe"
        wartosci.extend(list(set(X[:, i])))
        data_type[i] = (typ, wartosci)

    if len(y) != 0:
        classifier_classes = list(set(y))

        #sprawdzenie czy w wektorze y znajduja sie tylko dwie klasy
        if not len(classifier_classes) <= 2:
            raise ValueError

    if len(y) != 0:
        return data_type, classifier_classes
    else:
        return data_type