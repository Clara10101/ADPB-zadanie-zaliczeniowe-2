import unittest
import RandomForest
from RandomForest.BinTree import analyse_input_data, BinTree, is_numeric
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

    y = np.array(['KUP', 'NIE_KUPUJ', 'NIE_KUPUJ','KUP', 'NIE_KUPUJ', 'NIE_KUPUJ', 'KUP'])

    z = np.array([180000, 10100, 215000, 130000, 200000, 13000, 15000])

    data_type, classifier_classes = analyse_input_data(X,y)

    def test_find_best_division_1(self):
        "Test dla klasyfikacji"

        tree = BinTree(3,self.X,self.y,self.data_type,self.classifier_classes)
        random.seed(99)

        actual = tree.root().find_best_division(3,self.X,self.y,which='C')
        expected = ((1, '2006', 'numeryczne'), [1, 2, 5], [0, 3, 4, 6])

        self.assertEqual(expected, actual)

    def test_find_best_division_2(self):
        "Test dla regresji"

        tree = BinTree(3,self.X,self.z,self.data_type,[],which='R')
        random.seed(99)

        actual = tree.root().find_best_division(3,self.X,self.z,'R')
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


if __name__ == '__main__':
    unittest.main()

