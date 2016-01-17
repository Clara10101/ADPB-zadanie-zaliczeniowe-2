from RandomForestClassifier import *

class RandomForestRegressor(RandomForestBase):

    def __init__(self, n_features):

        super(RandomForestRegressor, self).__init__(n_features)

    def fit(self, X, y):
        """
        Metoda do nauki klasyfikatora na zbiorze X.
        :param X: Zbior treningowy X.
        :param y: Wektor, ktory dla kazdego wiersza X zawiera wartosc zmiennej zaleznej.
        """

        m, n = X.shape

        data_type = analyse_input_data(X)
        self.training_data_type = data_type

        #sprawdzenie czy wektor y zawiera wartosci nienumeryczne i czy ma odpowiednia dlugosc
        if not (all(map(is_numeric , y)) or len(y) != m):
            raise ValueError

        new_tree = True
        last_oob_errors = deque(maxlen=11)

        #decyzje podjete przez kazde drzewo dla kazdego wiersza obserwacji
        decisions = [[] for i in range(m)]

        #srednia wartosc y
        mean_y = np.mean(y)

        while new_tree:

            #losowanie ze zwracaniem m przykladow ze zbioru treningowego
            rows = np.random.choice(m, m, replace=True)
            training_set_X = X[rows,:]
            training_set_y = y[rows]

            #wiersze ktore nie sa w zbiorze treningowym - out of bag
            not_rows = list(set(range(m)).difference(rows))
            testing_set_X = X[not_rows,:]
            testing_set_y = y[not_rows]

            tree = self.create_decision_tree(training_set_X, training_set_y, which='R')
            self.forest.append(tree)

            #dla kazdej obserwacji sprawdzamy decyzje utworzone przez dodane drzewo
            for i in not_rows:
                decisions[i].append(tree.classify(X[i, :]))

            #r2 jako warunek zakonczenia uczenia lasow losowych dla regresji
            r2_score = sum((np.mean(decisions[i]) - mean_y)**2 for i in range(m) if decisions[i]) / float(sum((y[i] - mean_y)**2 for i in range(m)))
            last_oob_errors.append(r2_score)

            #sprawdzenie czy mozna zakonczyc proces uczenia nowych drzew
            #wyliczenie bledu oob
            if len(last_oob_errors) == 11:
                if list(last_oob_errors)[0] - (sum(list(last_oob_errors)[1:]) / 10.) < 0.001:
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
        """
        Zwraca liste zawierajaca decyzje podjete przez kazde drzewo w lesie.
        """

        m, n = X.shape

        #decyzje dla kazdego drzewa w lesie
        decisions = [[] for i in range(m)]
        for i,row in enumerate(X):
            for tree in self.forest:
                decision = tree.classify(row)
                decisions[i].append(decision)

        return decisions