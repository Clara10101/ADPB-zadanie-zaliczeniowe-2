import unittest
import RandomForest
from RandomForest.BinTree import analyse_input_data
import numpy as np


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


if __name__ == '__main__':
    unittest.main()

