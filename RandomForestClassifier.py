import numpy as np
import random
from collections import deque, Counter

inf = float('inf')

class BinNode:
    """
    Klasa reprezentujaca wezel w drzewie binarnym.
    """

    def __init__(self, values, data_type, classifier_classes, left=None, right=None, condition=None, decision=None):
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

    def set_sons_values(self, n_features, X, y):
        """
        Przypisanie rekurencyjnie obserwacji do synow korzenia na podstawie wybranego najlepszego kryterium podzialu
        :param n_features: liczba calkowita wieksza od 1 i mniejsza od liczby wszystkich obserwacji w danych treningowych
        :param X: dane treningowe, tablica numpy array, w ktorej kazdy wiersz odpowiada jednej obserwacji
        :param y: wektor klasyfikacji dla danych treningowych, numpy array
        """

        condition = self.find_best_division(n_features, X, y)

        #jesli podzial jest mozliwy i potrzebny
        if condition[0] != None and len(set(self.values_classes(y))) > 1:

            self.condition = condition[0]
            self.left = BinNode(condition[1], self.training_data_type, self.classifier_classes)
            self.right = BinNode(condition[2], self.training_data_type, self.classifier_classes)

            self.left.set_sons_values(n_features, X, y)
            self.right.set_sons_values(n_features, X, y)

        if len(set(self.values_classes(y))) == 1:
            self.decision = list(set(self.values_classes(y)))[0]

    def values_classes(self, y):
        return y[self.values]

    def find_best_division(self, n_features, X, y):
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

        random_features = random.sample(range(0, n), n_features)

        best_division_condition = None; best_L_values = None; best_R_values = None;
        best_gini_impurity = inf
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
                n_L0 = sum([val == classifier_classes[0] for val in y[L_values]])
                n_L1 = n_all - n_L0
                n_R0 = sum([val == classifier_classes[0] for val in y[R_values]])
                n_R1 = n_all - n_R0

                #interesuje nas tylko podzial dzie kazdy z synow ma przypisane jakies obserwacje
                if n_L != 0 and n_R != 0:
                    gini_impurity = gini(n_all,n_L,n_R,n_L0,n_L1,n_R0,n_R1)

                    #sprawdzenie czy do tej pory najlepszy warunek - minimalizuje gini
                    if gini_impurity < best_gini_impurity:
                        best_gini_impurity = gini_impurity
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

    def __init__(self, n_features, X, y, data_type, classifier_classes):

        m, n = X.shape
        self.n_features = n_features
        self.X = X
        self.y = y

        #typ danych na ktorych uczony byl klasyfikator tworzacy dane drzewo
        self.training_data_type = data_type
        self.classifier_classes = classifier_classes

        self.node = BinNode([i for i in range(m)],self.training_data_type, self.classifier_classes)
        self.node.set_sons_values(n_features, X, y)

    def root(self):
        return self.node

    def classify(self,v):
        return self.root().classify(v)



class RandomForestClassifier:
    """
    Klasa wykonujaca klasyfikacje za pomoca lasu losowego.
    """

    def __init__(self, n_features):

        self.n_features = n_features
        self.forest = [] #tablica zawierajaca drzewa klasyfikujace
        self.classifier_classes = [] #klasy klasyfikacji
        self.training_data_type = [] #typ danych i przyjmowane wartosci dla kazdej kolumny

    def fit(self, X, y):

        m, n = X.shape

        data_type, classifier_classes = analyse_input_data(X, y)
        self.training_data_type = data_type

        new_tree = True
        last_oob_errors = deque(maxlen=11)

        #zakladamy ze tylko dwie klasy
        #jako pierwsza w liscie ma byc klasa y[0] - wystepujaca w zbiorze treneigowym jako pierwsza
        self.classifier_classes = [y[0]]
        for c in classifier_classes:
            if c != y[0]:
                self.classifier_classes.append(c)

        #klasyfikacje dokonane za pomoca k-1 drzew
        actual_classification_for_training_set = [Counter({self.classifier_classes[0] : 0, self.classifier_classes[1] : 0}) for i in range(m)]

        #proporcje pomiedzy klasamy
        prop = Counter(y)[self.classifier_classes[0]] / float(m)

        while new_tree:

            losuj = True #zmienna odpowiadajaca za losowanie - sprawdzenie proporcji pomiedzy klasami

            while losuj:
                #losowanie ze zwracaniem m przykladow ze zbioru treningowego
                rows = np.random.choice(m, m, replace=True)
                training_set = X[rows,:]
                training_set_classes = y[rows]

                #sprawdzenie czy podobne proporcje pomiedzy klasami do tych w pelnym zbiorze treningowym
                act_prop = Counter(training_set_classes)[self.classifier_classes[0]] / float(m)

                if prop - 0.2 <= act_prop <= prop + 0.2:
                    losuj = False

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

    def check_predict_data(self,X):
        """
        Sprawdzenie poprawnosci danych na ktorych przeprowadzana jest klasyfikacja
        """

        pred_data_type = analyse_input_data(X)

        #sprawdzenie czy drugi wyiar jest rowny n
        if X.shape[1] != len(self.training_data_type):
            raise ValueError

        for i in range(len(pred_data_type)):
            #sprawdzenie czy typy kolumn zgadzaja sie z danymi treningowymi
            if pred_data_type[i][0] != self.training_data_type[i][0]:
                return ValueError
            #sprawdzenie czy w kolumnie wyliczeniowej podano wartosc, ktora nie wystepowala w tej kolumnie w zbiorze uczacym
            if pred_data_type[i][0] == 'wyliczeniowe':
                for j in pred_data_type[i][1]:
                    if j not in self.training_data_type[i][1]:
                        return ValueError

    def predict(self,X):

        self.check_predict_data(X)

        decisions = self.decisions_made_by_all_trees(X)
        forest_decisions = []

        for dec in decisions:
            forest_decisions.append(dec.most_common(1)[0][0])

        return forest_decisions

    def predict_proba(self,X):
        """
        Zwraca wektor prawdopodobienstw przynaleznosci przykladow do pierwszej klasy - czyli do klasy wystepujacej w zbiorze treningowym jako pierwsza
        """

        self.check_predict_data(X)

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
        """
        Tworzy drzewo decyzyjne.
        """

        tree = BinTree(self.n_features, X, y, self.training_data_type, self.classifier_classes)
        return tree



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


def is_numeric(x):
    return x.replace('.', '', 1).isdigit()

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

def showR(node, prefix=''):
    """
    Rysuje w sposob rekurencyjny drzewo.
    """

    if node.is_leaf():
        return prefix + '-' + str(node) + '\n'
    else:
        return showR(node.son('L'),prefix+'   ') + prefix + '-<' + '\n' + showR(node.son('R'),prefix+'   ')

def show(tree):
    return showR(tree.root())



# Dane testowe
dane_test_X = np.array(
    [['Honda', 2009, 'igla', 180000.87], ['Honda', 2005, 'igla', 10100], ['Honda', 2006, 'idealny', 215000], ['Renault', 2010, 'igla', 130000], ['Renault', 2007, 'idealny', 200000]])
dane_test_y = np.array(['KUP', 'NIE_KUPUJ', 'NIE_KUPUJ','KUP', 'NIE_KUPUJ'])

#analyse_input_data(dane_test_X, dane_test_y)
#przykladowy wynik wywolania
#[('wyliczeniowe', ['Honda']), ('numeryczne', ['2009', '2006', '2005']), ('wyliczeniowe', ['igla', 'idealny']), ('numeryczne', ['180000.87', '10100', '215000'])]

r = RandomForestClassifier(3)
#tree = r.create_decision_tree(dane_test_X, dane_test_y)
#print tree.root()
#print show(tree)

przyklad_testowy = ['Renault', 2005, 'bezkolizyjny', 215000]
#print tree.classify(przyklad_testowy)

r.fit(dane_test_X,dane_test_y)
print r.forest
print r.predict(dane_test_X)
print r.predict_proba(dane_test_X)