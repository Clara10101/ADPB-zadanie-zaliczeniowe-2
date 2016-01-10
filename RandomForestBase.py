from abc import ABCMeta, abstractmethod
from BinTree import *

class RandomForestBase:
    """
    Klasa bazowa dla klasyfikacji za pomoca lasu losowego.
    """
    __metaclass__ = ABCMeta

    def __init__(self, n_features):
        self.n_features = n_features
        self.forest = [] #tablica zawierajaca drzewa klasyfikujace
        self.training_data_type = [] #typ danych i przyjmowane wartosci dla kazdej kolumny

    def check_predict_data(self,X):
        """
        Sprawdzenie poprawnosci danych dla ktorych wywolywana jest metoda predict
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

    @abstractmethod
    def fit(self, X, y):
        """
        Metoda do nauki klasyfikatora na zbiorze treningowym X.
        """

    @abstractmethod
    def predict(self, X):
        """
        Zwraca wynik klasyfikacji X.
        """

    @abstractmethod
    def create_decision_tree(self, X, y, which):
        """
        Tworzy pojedyncze drzewo decyzyjne.
        """

    @abstractmethod
    def decisions_made_by_all_trees(self,X):
        """
        Zwraca liste zawierajaca decyzje podjete przez kazde drzewo w lesie.
        """
