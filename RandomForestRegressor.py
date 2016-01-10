from RandomForestClassifier import *

class RandomForestRegressor(RandomForestBase):

    def __init__(self, n_features):

        super(RandomForestClassifier, self).__init__(n_features)

    def fit(self, X, y):
        """
        Metoda do nauki klasyfikatora na zbiorze X.
        :param X: Zbior treningowy X.
        :param y: Wektor, ktory dla kazdego wiersza X zawiera wartosc zmiennej zaleznej.
        """

        m, n = X.shape

        #sprawdzenie czy wektor y zawiera wartosci nienumeryczne i czy ma odpowiednia dlugosc
        if not (all(map(is_numeric , y)) or len(y) != m):
            raise ValueError

        new_tree = True
        last_oob_errors = deque(maxlen=11)

        while new_tree:

            #losowanie ze zwracaniem m przykladow ze zbioru treningowego
            rows = np.random.choice(m, m, replace=True)
            training_set_X = X[rows,:]
            training_set_y = y[rows]

            #wiersze ktore nie sa w zbiorze treningowym - out of bag
            not_rows = list(set(range(m)).difference(rows))
            testing_set_X = X[not_rows,:]
            testing_set_y = y[not_rows]

            tree = self.create_decision_tree(training_set_X, training_set_y, 'R')
            self.forest.append(tree)

            #wartosci przypisane przez aktualne drzewo
            decisions = []

            #dla kazdej obserwacji sprawdzamy decyzje utworzone przez dodane drzewo
            for row in testing_set_X:
                decisions.append(tree.classify(row))

            ###Poprawic warunek stopu dla tworzenia nowych drzew i obliczanie r2

            mean_y = np.mean(testing_set_y)
            r2_score = sum(decisions[i] - mean_y for i in range(len(testing_set_y)))**2 / float(sum(testing_set_y[i] - mean_y for i in range(len(testing_set_y)))**2)

            last_oob_errors.append(r2_score)

            #sprawdzenie czy mozna zakonczyc proces uczenia nowych drzew
            #wyliczenie bledu oob
            if len(last_oob_errors) == 11:
                if list(last_oob_errors)[0] - (sum(list(last_oob_errors)[1:]) / 10.) < 0.01:
                    new_tree = False

    def predict(self, X):
        """
        Wyznacza wynik regresji dla przykladow w X; zwraca wektor liczb rzeczywistych
        """

        self.check_predict_data(X)
        decisions = self.decisions_made_by_all_trees(X)
        forest_decisions = []

        for dec in decisions:
            #srednia z decyzji wszystkich drzew
            forest_decisions.append(np.mean(dec))

        return forest_decisions

    def decisions_made_by_all_trees(self,X):

        m, n = X.shape

        #decyzje dla kazdego drzewa w lesie
        decisions = [[] for i in range(m)]
        for i,row in enumerate(X):
            for tree in self.forest:
                decision = tree.classify(row)
                decisions[i].append(decision)

        return decisions