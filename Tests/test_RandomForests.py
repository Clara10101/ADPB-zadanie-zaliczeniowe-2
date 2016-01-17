import unittest
import RandomForest
from RandomForest.BinTree import analyse_input_data, BinNode, BinTree, is_numeric
import numpy as np
import random


class TestBinTreeAnalyseInputData(unittest.TestCase):
    "Test dla funkcji analyse_input_data sprawdzajacej typ danych wejsciowych"

    def test_analyse_input_data_1(self):
        "Test dla poprawnych danych wejsciowych, bez podanego wektora klas"

        actual = analyse_input_data(np.array(
            [['Honda', 2009, 'igla', 180000.87],
             ['Honda', 2005, 'igla', 10100],
             ['Honda', 2006, 'idealny', 215000]]))
        expected = [('wyliczeniowe', ['Honda']), ('numeryczne', ['2009', '2006', '2005']),
                    ('wyliczeniowe', ['igla', 'idealny']), ('numeryczne', ['180000.87', '10100', '215000'])]
        self.assertEqual(expected, actual)

    def test_analyse_input_data_2(self):
        "Poprawde dane wejsciowe, dodatkowo z podanym wektorem klas"

        actual = analyse_input_data(np.array(
            [['Honda', 2006, 'idealny', 215000],
             ['Renault', 2010, 'igla', 130000],
             ['Renault', 2007, 'idealny', 200000]]), np.array(['NIE_KUPUJ', 'KUP', 'NIE_KUPUJ'])
        )
        expected = ([('wyliczeniowe', ['Honda', 'Renault']), ('numeryczne', ['2006', '2010', '2007']),
                     ('wyliczeniowe', ['igla', 'idealny']), ('numeryczne', ['200000', '130000', '215000'])],
                    ['NIE_KUPUJ', 'KUP'])
        self.assertEqual(expected, actual)

    def test_analyse_input_data_3(self):
        "Bledne dane wejsciowe w postaci wektora klas o dlugosci innej niz liczba wierszy macierzy danych"

        self.assertRaises(ValueError, analyse_input_data, np.array(
            [['Honda', 2006, 'idealny', 215000],
             ['Renault', 2010, 'igla', 130000]]), np.array(['NIE_KUPUJ', 'KUP', 'NIE_KUPUJ', 'NIE_KUPUJ']))

    def test_analyse_input_data_4(self):
        "Bledne dane wejsciowe, wektor klas zawiera wiecej niz dwie klasy"

        self.assertRaises(ValueError, analyse_input_data, np.array(
            [['Honda', 2006, 'idealny', 215000],
             ['Renault', 2010, 'igla', 130000]]), np.array(['NIE_KUPUJ', 'KUP', 'NIE_KUPUJ', 'NIE_KUPUJ', 'NIE_WIEM']))


class TestBinTreeFindBestDivision(unittest.TestCase):
    "Test dla funkcji znajdujacej optymalny podzial w wezle"

    # testowe dane wejsciowe
    X = np.array(
        [['Honda', 2009, 'igla', 180000.87],
         ['Honda', 2005, 'igla', 10100],
         ['Honda', 2006, 'idealny', 215000],
         ['Renault', 2010, 'igla', 130000],
         ['Renault', 2007, 'idealny', 200000],
         ['Renault', 2005, 'bezkolizyjny', 215000],
         ['Ford', 2008, 'bezkolizyjny', 225000]])

    y = np.array(['KUP', 'NIE_KUPUJ', 'NIE_KUPUJ', 'KUP', 'NIE_KUPUJ', 'NIE_KUPUJ', 'KUP'])

    z = np.array([180000, 10100, 215000, 130000, 200000, 13000, 15000])

    data_type, classifier_classes = analyse_input_data(X, y)

    def test_find_best_division_1(self):
        "Test dla klasyfikacji"

        tree = BinTree(3, self.X, self.y, self.data_type, self.classifier_classes)
        random.seed(99)

        actual = tree.root().find_best_division(3, self.X, self.y, which='C')
        expected = ((1, '2006', 'numeryczne'), [1, 2, 5], [0, 3, 4, 6])

        self.assertEqual(expected, actual)

    def test_find_best_division_2(self):
        "Test dla regresji"

        tree = BinTree(3, self.X, self.z, self.data_type, [], which='R')
        random.seed(99)

        actual = tree.root().find_best_division(3, self.X, self.z, 'R')
        expected = ((2, 'idealny', 'wyliczeniowe'), [2, 4], [0, 1, 3, 5, 6])

        self.assertEqual(expected, actual)


class BinTreeIsNumeric(unittest.TestCase):
    "Test dla funkcji sprawdzajacej czy jej argument jest wartoscia numeryczna (int, float)"

    def test_is_numeric_1(self):
        "Test dla float"

        actual = is_numeric(float(1.2))
        expected = True

        self.assertEqual(expected, actual)

    def test_is_numeric_2(self):
        "Test dla int"

        actual = is_numeric(int(5))
        expected = True

        self.assertEqual(expected, actual)

    def test_is_numeric_3(self):
        "Test dla str"

        actual = is_numeric('abc')
        expected = False

        self.assertEqual(expected, actual)


class TestBinTreeIsLeaf(unittest.TestCase):
    "Test dla funkcji is_leaf sprawdzajacej czy wezel jest lisciem"

    def test_is_leaf_1(self):
        "Test dla wezla bedacego lisciem"

        node = BinNode(5, int)
        actual = node.is_leaf()
        expected = True
        self.assertEqual(expected, actual)

    def test_is_leaf_2(self):
        "Test dla wezla nie bedacego lisciem, majacego prawego i lewego syna"

        node = BinNode(5, int, left=4, right=3)
        actual = node.is_leaf()
        expected = False
        self.assertEqual(expected, actual)

    def test_is_leaf_3(self):
        "Test dla wezla nie bedacego lisciem, majacego prawego syna"

        node = BinNode(5, int, right=3)
        actual = node.is_leaf()
        expected = False
        self.assertEqual(expected, actual)


class TestSon(unittest.TestCase):
    "Test dla funkcji son zwracajacego wybranego syna wezla: prawego lub lewego"

    def test_is_leaf_1(self):
        "Test dla wezla posiadajacego dwoch synow, zwracanie lewego syna"

        node = BinNode(5, int, left=2, right=1)
        actual = node.son('L')
        expected = 2
        self.assertEqual(expected, actual)

    def test_is_leaf_2(self):
        "Test dla wezla posiadajacego dwoch synow, zwracanie prawego syna"

        node = BinNode(5, int, left=2, right=1)
        actual = node.son('R')
        expected = 1
        self.assertEqual(expected, actual)

    def test_is_leaf_3(self):
        "Test dla wezla nie posiadajacego zadnego syna"

        node = BinNode(5, int)
        actual = node.son('R'), node.son('L')
        expected = None, None
        self.assertEqual(expected, actual)


class TestSetSonsValues(unittest.TestCase):
    "Test dla funkcji przypisujacej obserwacje z rodzica do odpowiedniego dziecka"

    X = np.array(
        [['Honda', 2009, 'igla', 180000.87],
         ['Honda', 2005, 'igla', 10100],
         ['Honda', 2006, 'idealny', 215000],
         ['Renault', 2010, 'igla', 130000]])

    y = np.array(['KUP', 'NIE_KUPUJ','NIE_KUPUJ', 'KUP'])

    z = np.array([180000, 10100, 215000, 130000])

    data_type, classifier_classes = analyse_input_data(X, y)

    def test_set_sons_values_1(self):
        node = BinNode([0, 1], self.data_type, self.classifier_classes)
        random.seed(99)

        # przed wywolaniem metody set_sons_values lewy i prawy syn wierzcholka powinny byc None
        left_son_before = node.son("L")
        self.assertIsNone(left_son_before)
        right_son_before = node.son("R")
        self.assertIsNone(right_son_before)

        #po wywolaniu metody powinny byc to wierzcholki
        node.set_sons_values(3, self.X, self.y, which='C')

        self.assertIsInstance(node.son("L"), BinNode)
        self.assertIsInstance(node.son("R"), BinNode)

        actual_left_son_values = node.son("L").values
        actual_right_son_values = node.son("R").values

        expected_left_son_values = [1]
        expected_right_son_values = [0]

        self.assertEqual(actual_left_son_values, expected_left_son_values)
        self.assertEqual(actual_right_son_values, expected_right_son_values)

        actual_left_son_decision = node.son("L").decision
        actual_right_son_decision = node.son("R").decision

        expected_left_son_decision = 'NIE_KUPUJ'
        expected_right_son_decision = 'KUP'

        self.assertEqual(actual_left_son_decision, expected_left_son_decision)
        self.assertEqual(actual_right_son_decision, expected_right_son_decision)

    def test_set_sons_values_2(self):
        "Test dla wierzcholka zawierajacego tylko jedna obserwacje z jednej klasy"

        #Taki wierzcholek jest lisciem i metoda nie powinna tworzyc dzieci
        node = BinNode([0], self.data_type, self.classifier_classes)
        random.seed(99)

        #przed wywolaniem metody set_sons_values lewy i prawy syn wierzcholka powinny byc None
        left_son = node.son("L")
        self.assertIsNone(left_son)
        right_son = node.son("R")
        self.assertIsNone(right_son)

        node.set_sons_values(3, self.X, self.y, which='C')

        self.assertIsNone(left_son)
        self.assertIsNone(right_son)

    def test_set_sons_values_3(self):
        "Test dla regresji i wierzcholka zawierajacego 4 obserwacje"

        node = BinNode([0,1,2,3], self.data_type)

        left_son_before = node.son("L")
        right_son_before = node.son("R")

        #Przed wywolaniem fukcji dzieci powinny byc None
        self.assertIsNone(left_son_before)
        self.assertIsNone(right_son_before)

        node.set_sons_values(3, self.X, self.z, which='R')

        left_son_after = node.son("L")
        right_son_after = node.son("R")

        self.assertEqual(left_son_after.values,[2])
        self.assertEqual(right_son_after.values,[0,1,3])

class TestTree(unittest.TestCase):
    def test_create_tree(self):
        "Test sprawdzajacy tworzenie drzewa"

        n_features, X, y, data_type, classifier_classes = 1, np.array([[1, 1], [2, 2]]), \
                                                          np.array([1, 2]), [('numeryczne', [1, 2])], 'R'
        tree = BinTree(n_features, X, y, data_type, classifier_classes)
        self.assertIsInstance(tree.root(), BinNode)
        self.assertEquals(tree.node.left.is_leaf(), True)
        self.assertEquals(tree.node.right.is_leaf(), True)

		


if __name__ == '__main__':
    unittest.main()

