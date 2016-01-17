import unittest
import RandomForest
from RandomForest.BinTree import analyse_input_data, BinNode
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

class TestBinTreeIsLeaf(unittest.TestCase):
    "Test dla funkcji is_leaf sprawdzajacej czy wezel jest lisciem"

    def test_is_leaf_1(self):
        "Test dla wezla bedacego lisciem"

        node=BinNode(5,int)
        actual = node.is_leaf()
        expected = True
        self.assertEqual(expected, actual)

    def test_is_leaf_2(self):
        "Test dla wezla nie bedacego lisciem, majacego prawego i lewego syna"

        node=BinNode(5,int,left=4,right=3)
        actual = node.is_leaf()
        expected = False
        self.assertEqual(expected, actual)

    def test_is_leaf_2(self):
        "Test dla wezla nie bedacego lisciem, majacego prawego syna"

        node=BinNode(5,int,right=3)
        actual = node.is_leaf()
        expected = False
        self.assertEqual(expected, actual)
        
class TestSon(unittest.TestCase):
    "Test dla funkcji son zwracajacego wybranego syna wezla: prawego lub lewego"

    def test_is_leaf_1(self):
        "Test dla wezla posiadajacego dwoch synow, zwracanie lewego syna"

        node=BinNode(5,int,left=2,right=1)
        actual = node.son('L')
        expected = 2
        self.assertEqual(expected, actual)

    def test_is_leaf_2(self):
        "Test dla wezla posiadajacego dwoch synow, zwracanie prawego syna"

        node=BinNode(5,int,left=2,right=1)
        actual = node.son('R')
        expected = 1
        self.assertEqual(expected, actual)

    def test_is_leaf_2(self):
        "Test dla wezla nie posiadajacego zadnego syna"

        node=BinNode(5,int)
        actual = node.son('R'),node.son('L')
        expected = None,None
        self.assertEqual(expected, actual)

class TestTree(unittest.TestCase):
		
    def test_create_tree(self):
        "Test sprawdzajacy tworzenie drzewa"
		
        n_features, X, y, data_type, classifier_classes = 1, np.array([[1,1],[2,2]]),\
                                                          np.array([1,2]), [('numeryczne',[1,2])], 'R'
        tree = BinTree(n_features, X, y, data_type, classifier_classes)
        self.assertIsInstance(tree.root(), BinNode)
        self.assertEquals(tree.node.left.is_leaf(), True)
        self.assertEquals(tree.node.right.is_leaf(), True)
		
if __name__ == '__main__':
    unittest.main()

