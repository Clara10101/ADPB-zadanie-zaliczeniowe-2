import numpy as np
import random
from collections import deque, Counter
from BinTree import *


class RandomForestClassifier:
    """
    Klasa wykonujaca klasyfikacje za pomoca lasu losowego.
    """
    #w create_decision_tree jest obecnie which='C' dlatego bedzie to tylko dla klasyfikacji

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

        tree = BinTree(self.n_features, X, y, self.training_data_type, self.classifier_classes, which='C')

        return tree


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