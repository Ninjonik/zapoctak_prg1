"""
Jednoduchá maticová kalkulačka na lineárnu algebru.
Zápočtový program, zimný semester 2025/2026, Programovanie 1.

Peter Zaťko

Testovacia sada

Pre spustenie testovacej sady:
python -m unittest test_matrix -v

"""

import unittest
from typing import List

import numpy as np
from main import Matrix


def matrix_to_numpy(matrix):
    """Konverzuje objekt matice na numpy pole."""
    if matrix.empty:
        return np.array([])
    return np.array([list(matrix[i]) for i in range(1, matrix.rows + 1)])


def numpy_to_matrix(array):
    """Konvertuje numpy pole na objekt matice."""
    if array.size == 0:
        return Matrix([])
    return Matrix(array.tolist())


class TestMatrixConstruction(unittest.TestCase):
    """Základná testovacia sada - testuje konštrukciu matice na rôznych vstupoch."""

    def test_init_from_list(self):
        """Vytvorenie matice z poľa."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 3)
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 3], 6)

    def test_zeros(self):
        """Vytvorenie nulovej matice."""
        m = Matrix.zeros(3, 4)
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 4)
        for i in range(1, 4):
            for j in range(1, 5):
                self.assertEqual(m[i, j], 0)

    def test_identity(self):
        """Vytvorenie jednotkovej matice."""
        m = Matrix.identity(3)
        for i in range(1, 4):
            for j in range(1, 4):
                self.assertEqual(m[i, j], 1 if i == j else 0)

    def test_from_list(self):
        """Vytvorenie matice z poľa (explicitný spôsob)."""
        m = Matrix.from_list([[1, 2], [3, 4]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m[1, 1], 1)

    def test_from_matrix(self):
        """Vytvorenie matice z inej matice (kópia)."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix.from_matrix(m1)
        self.assertEqual(m1, m2)
        # Test, či sme len nevytvorili ďalšiu referenciu, ale či je m2 skutočne iný objekt.
        m2[1, 1] = 99
        self.assertNotEqual(m1[1, 1], m2[1, 1])

    def test_empty_matrix(self):
        """Test vytvorenia prázdnej matice."""
        m = Matrix([])
        self.assertTrue(m.empty)
        self.assertEqual(m.rows, 0)
        self.assertEqual(m.cols, 0)


class TestMatrixProperties(unittest.TestCase):
    """Testuje základné vlastnosti matice."""

    def test_shape(self):
        """Testuje správnosť rádu matice."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.shape, (2, 3))

    def test_rows_cols(self):
        """Testuje správnosť počtu riadkov a počtu stĺpcov."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 3)

    def test_is_square(self):
        """Testuje správnosť štvorcovitosti."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(m1.is_square)
        self.assertFalse(m2.is_square)

    def test_is_symmetric(self):
        """Testuje správnosť symetričnosti."""
        symmetric = Matrix([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        not_symmetric = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        not_square = Matrix([[1, 2, 3], [4, 5, 6]])

        self.assertTrue(symmetric.is_symmetric)
        self.assertFalse(not_symmetric.is_symmetric)
        self.assertFalse(not_square.is_symmetric)

    def test_empty_property(self):
        """Testuje správnosť prázdnosti."""
        m1 = Matrix([])
        m2 = Matrix([[1, 2]])
        self.assertTrue(m1.empty)
        self.assertFalse(m2.empty)


class TestMatrixIndexing(unittest.TestCase):
    """Testuje pristupovanie k prvkom v matici."""

    def test_getitem_row(self):
        """Testuje prístup k riadku matice (ku riadku)."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        row1 = m[1]
        row2 = m[2]
        self.assertEqual(row1, (1, 2, 3))
        self.assertEqual(row2, (4, 5, 6))

    def test_getitem_element(self):
        """Testuje prístup k riadku a stĺpcu matice (k jednému prvku)."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 3], 3)
        self.assertEqual(m[2, 2], 5)

    def test_getitem_out_of_range(self):
        """Testuje, či matica správne vráti chybu pri pokuse pristupovať k nevalidným prvkom."""
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(IndexError):
            _ = m[3]
        with self.assertRaises(IndexError):
            _ = m[1, 3]
        with self.assertRaises(IndexError):
            _ = m[0]

    def test_setitem_row(self):
        """Testuje nastavovanie riadku na určitú hodnotu."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m[1] = [7, 8, 9]
        self.assertEqual(m[1], (7, 8, 9))

    def test_setitem_element(self):
        """Testuje nastavovanie prvku na určitú hodnotu."""
        m = Matrix([[1, 2], [3, 4]])
        m[1, 1] = 99
        self.assertEqual(m[1, 1], 99)

    def test_setitem_wrong_length(self):
        """Testuje nastaviť riadok na iný riadok s nesprávnym rozmerom."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m[1] = [1, 2]

    def test_iteration(self):
        """Testuje iterovateľnosť matice."""
        m = Matrix([[1, 2], [3, 4]])
        rows = list(m)
        self.assertEqual(rows, [(1, 2), (3, 4)])


class TestMatrixEquality(unittest.TestCase):
    """Testuje porovnávanie rovnosti matíc."""

    def test_equal_matrices(self):
        """Keď sa matice rovnajú."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m1, m2)

    def test_unequal_matrices(self):
        """Keď sa matice nerovnajú."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 5]])
        self.assertNotEqual(m1, m2)

    def test_different_shapes(self):
        """Keď majú matice rôzne rozmery."""
        m1 = Matrix([[1, 2]])
        m2 = Matrix([[1], [2]])
        self.assertNotEqual(m1, m2)

    def test_empty_matrices(self):
        """Keď sú matice prázdne."""
        m1 = Matrix([])
        m2 = Matrix([])
        self.assertEqual(m1, m2)

    def test_empty_vs_none(self):
        """Keď porovnávame prázdnu maticu s None."""
        m = Matrix([])
        self.assertEqual(m, None)


class TestMatrixCopy(unittest.TestCase):
    """Testuje operáciu kopírovania matice."""

    def test_copy(self):
        """Testuje kopírovanie matice."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = m1.copy()
        self.assertEqual(m1, m2)
        m2[1, 1] = 99
        self.assertNotEqual(m1[1, 1], m2[1, 1])


class TestMatrixTranspose(unittest.TestCase):
    """Testuje operáciu transpozície."""

    def test_transpose_rectangular(self):
        """Testuje transpozíciu obdĺžnikovej matice."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        np_m = matrix_to_numpy(m)

        # Výsledok, ktorý predpokladáme ako správny
        expected_result = np.transpose(np_m)

        m.transpose()

        # Konvertujeme na typ numpy pre jednoduché porovnanie
        np_result = matrix_to_numpy(m)

        np.testing.assert_array_equal(np_result, expected_result)

        # Ďalšie testy
        self.assertEqual(m.shape, (3, 2))
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 4)
        self.assertEqual(m[3, 1], 3)
        self.assertEqual(m[3, 2], 6)

    def test_transpose_square(self):
        """Testuje transpozíciu štvorcovej matice."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        np_m = matrix_to_numpy(m)

        # Výsledok, ktorý predpokladáme ako správny
        expected_result = np.transpose(np_m)

        m.transpose()

        # Konvertujeme na typ numpy pre jednoduché porovnanie
        np_result = matrix_to_numpy(m)

        np.testing.assert_array_equal(np_result, expected_result)

        # Ďalšie testy
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 4)
        self.assertEqual(m[1, 3], 7)
        self.assertEqual(m[2, 1], 2)

    def test_transpose_symmetric(self):
        """Transpozícia symetrickej matice."""
        m = Matrix([[1, 2], [2, 3]])
        original = m.copy()

        np_m = matrix_to_numpy(m)

        expected_result = np.transpose(np_m)

        m.transpose()

        np_result = matrix_to_numpy(m)

        np.testing.assert_array_equal(np_result, expected_result)

        # Transponovaná symetrická matica musí byť taká istá ako originálna (netransponovaná) matica
        self.assertEqual(m, original)


class TestElementaryRowOperations(unittest.TestCase):
    """Testuje elementárne riadkové operácie (ERO)."""

    def test_scale_row(self):
        """ERO - násobenie riadku nenulovým celým parametrom."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.scale_row(1, 2)
        self.assertEqual(m[1], (2, 4, 6))
        self.assertEqual(m[2], (4, 5, 6))

    def test_scale_row_by_fraction(self):
        """ERO - násobenie riadku nenulovým reálnym parametrom."""
        m = Matrix([[2, 4, 6], [1, 2, 3]])
        m.scale_row(1, 0.5)
        self.assertEqual(m[1], (1, 2, 3))

    def test_scale_row_by_zero(self):
        """NEERO - násobenie riadku nulovým parametrom (nemalo by fungovať)."""
        m = Matrix([[1, 2], [3, 4]])
        original = m.copy()
        m.scale_row(1, 0)
        self.assertEqual(m, original)

    def test_swap_rows(self):
        """ERO - výmena dvoch riadkov"""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.swap_rows(1, 2)
        self.assertEqual(m[1], (4, 5, 6))
        self.assertEqual(m[2], (1, 2, 3))

    def test_add_row(self):
        """ERO - pripočítanie jedného riadku k druhému."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.add_row(1, 2, 1)
        self.assertEqual(m[1], (5, 7, 9))
        self.assertEqual(m[2], (4, 5, 6))

    def test_add_row_with_multiple(self):
        """ERO - pripočítanie reálneho násobku jedného riadku k druhému."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.add_row(1, 2, 2)
        self.assertEqual(m[1], (9, 12, 15))

    def test_subtract_row(self):
        """ERO - odčítanie reálneho násobku jedného riadku od druhého."""
        m = Matrix([[5, 7, 9], [4, 5, 6]])
        m.subtract_row(1, 2, 1)
        self.assertEqual(m[1], (1, 2, 3))


class TestPivotOperations(unittest.TestCase):
    """Testuje operácie spojené s pivotom."""

    def test_get_pivot(self):
        """Testuje získanie indexu pivota v danom (nenulovom) riadku."""
        m = Matrix([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        self.assertEqual(m.get_pivot(1), 1)
        self.assertEqual(m.get_pivot(2), 2)
        self.assertEqual(m.get_pivot(3), 3)

    def test_get_pivot_zero_row(self):
        """Testuje získanie indexu pivota v nulovom riadku."""
        m = Matrix([[1, 2], [0, 0]])
        self.assertIsNone(m.get_pivot(2))

    def test_is_row_empty(self):
        """Testuje, či je riadok prázdny - prázdny <=> nemá v riadku žiadny pivot."""
        m = Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
        self.assertFalse(m.is_row_empty(1))
        self.assertTrue(m.is_row_empty(2))
        self.assertFalse(m.is_row_empty(3))


class TestRowRemoval(unittest.TestCase):
    """Testuje odstránenie riadku z matice."""

    def test_remove_row(self):
        """Testuje odstránenie nenulového riadku z matice."""
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        m.remove_row(2)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m[1], (1, 2))
        self.assertEqual(m[2], (5, 6))

    def test_remove_null_rows(self):
        """Testuje odstránenie nulového riadku"""
        m = Matrix([[1, 2], [0, 0], [3, 4], [0, 0]])
        m.remove_null_rows()
        self.assertEqual(m.rows, 2)
        self.assertEqual(m[1], (1, 2))
        self.assertEqual(m[2], (3, 4))

    def test_remove_null_rows_all_zero(self):
        """Testuje odstránenie riadku z nulovej matice."""
        m = Matrix([[0, 0], [0, 0]])
        m.remove_null_rows()
        self.assertEqual(m.rows, 0)
        self.assertTrue(m.empty)


class TestREF(unittest.TestCase):
    """Testuje operáciu Gaussovej eliminácie."""

    def test_ref_simple(self):
        """G. elim. na štandardnej matici, ktorá nie je v REF."""
        m = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        m.ref()

        last_pivot = -1
        for i in range(1, m.rows + 1):
            pivot = m.get_pivot(i)
            if pivot is not None:
                self.assertGreater(pivot, last_pivot)  # Pivot musí byť vpravo od toho predošlého
                for r_below in range(i + 1, m.rows + 1):
                    self.assertEqual(m[r_below, pivot], 0)  # Pod pivotom musia byť v riadkoch pod aktuálnym nuly
                last_pivot = pivot

    def test_ref_already_ref(self):
        """G. elim. na matici, ktorá už je v REF."""
        m1 = Matrix([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
        m2 = m1.copy()

        m1.ref()

        # Matica by mala byť úplne rovnaká.
        for i in range(1, m1.rows + 1):
            for j in range(1, m2.cols + 1):
                self.assertEqual(m1[i, j], m2[i, j])

    def test_ref_with_zero_row(self):
        """G. elim. na matici s nulovým riadkom."""
        m = Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]])

        m.ref()

        # Nulové riadky by mali byť eliminované.
        self.assertEqual(m.rows, 2)


class TestRREF(unittest.TestCase):
    """Testuje operáciu Gauss-Jordanovej eliminácie (RREF)."""

    def test_rref_simple(self):
        """GJ elim. na bežnej matici, ktorá nie je v RREF."""
        m = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        m.rref()

        last_pivot = -1
        for i in range(1, m.rows + 1):
            pivot = m.get_pivot(i)
            if pivot is not None:
                # Pivot musí byť vpravo od predchádzajúceho
                self.assertGreater(pivot, last_pivot)
                # Pivot musí byť rovný 1
                self.assertEqual(m[i, pivot], 1)
                # V stĺpci pivotu musia byť všade inde nuly
                for r in range(1, m.rows + 1):
                    if r != i:
                        self.assertEqual(m[r, pivot], 0)
                last_pivot = pivot

    def test_rref_already_rref(self):
        """GJ elim. na matici, ktorá už je v RREF."""
        m1 = Matrix([[1, 0, 2], [0, 1, 3]])
        m2 = m1.copy()

        m1.rref()

        # Matica by sa nemala zmeniť
        for i in range(1, m1.rows + 1):
            for j in range(1, m1.cols + 1):
                self.assertEqual(m1[i, j], m2[i, j])

    def test_rref_with_zero_row(self):
        """GJ elim. na matici s nulovým riadkom."""
        m = Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
        m.rref()

        # Nulové riadky by mali byť eliminované
        self.assertEqual(m.rows, 2)

    def test_rref_rectangular_more_rows(self):
        """GJ elim. na obdĺžnikovej matici (viac riadkov ako stĺpcov)."""
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        m.rref()

        # Prvé dva riadky majú tvoriť jednotkovú maticu
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[2, 1], 0)
        self.assertEqual(m[2, 2], 1)

    def test_rref_rectangular_more_cols(self):
        """GJ elim. na obdĺžnikovej matici (viac stĺpcov ako riadkov)."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.rref()

        # Pivoty majú byť v prvých dvoch stĺpcoch
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)

    def test_rref_singular_matrix(self):
        """GJ elim. na singulárnej matici."""
        m = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        m.rref()

        # Má ostať len 1 nenulový riadok (v tomto prípade)
        self.assertEqual(m.rows, 1)
        self.assertEqual(m[1, m.get_pivot(1)], 1)

    def test_rref_augmented_system(self):
        """GJ elim. na rozšírenej matici sústavy rovníc."""
        m = Matrix([[1, 2, 5], [3, 4, 11]])
        m.rref()

        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[1, 3], 1)
        self.assertEqual(m[2, 1], 0)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[2, 3], 2)

    def test_rref_single_row(self):
        """GJ elim. na matici s jedným riadkom."""
        m = Matrix([[2, 4, 6]])
        m.rref()

        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 2)
        self.assertEqual(m[1, 3], 3)

    def test_rref_single_column(self):
        """GJ elim. na matici s jedným stĺpcom."""
        m = Matrix([[2], [4], [6]])
        m.rref()

        self.assertEqual(m.rows, 1)
        self.assertEqual(m[1, 1], 1)

    def test_rref_all_zero(self):
        """GJ elim. na nulovej matici."""
        m = Matrix.zeros(3, 3)
        m.rref()

        # RREF nulovej matice by stále mala byť nulová matica
        self.assertTrue(m.empty)



class TestArithmeticOperations(unittest.TestCase):
    """Testuje aritmetické operácie."""

    def test_matrix_addition(self):
        """Testuje maticové sčítanie."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])

        np_m1 = matrix_to_numpy(m1)
        np_m2 = matrix_to_numpy(m2)

        expected_result = np_m1 + np_m2

        actual_result = m1 + m2

        np_actual = matrix_to_numpy(actual_result)

        # Porovnanie výsledkov
        np.testing.assert_array_equal(np_actual, expected_result)

        # Test komutativity maticového sčítania (malo by to byť to isté)
        actual_result_right = m2 + m1
        np_actual_right = matrix_to_numpy(actual_result_right)
        np.testing.assert_array_equal(np_actual_right, expected_result)

    def test_matrix_subtraction(self):
        """Test odčítania matíc."""
        m1 = Matrix([[10, 20], [30, 40]])
        m2 = Matrix([[5, 6], [7, 8]])

        np_m1 = matrix_to_numpy(m1)
        np_m2 = matrix_to_numpy(m2)

        expected_result = np_m1 - np_m2

        actual_result = m1 - m2

        np_actual = matrix_to_numpy(actual_result)

        # Porovnanie výsledkov
        np.testing.assert_array_equal(np_actual, expected_result)

    def test_scalar_multiplication(self):
        """Test skalárneho násobku matice."""
        m = Matrix([[1, 2], [3, 4]])
        scalar = 2

        np_m = matrix_to_numpy(m)

        expected_result = np_m * scalar

        actual_result = m * scalar

        np_actual = matrix_to_numpy(actual_result)

        # Porovnanie výsledkov
        np.testing.assert_array_equal(np_actual, expected_result)

        # Test skalárneho násobku sprava (nie matematicky korektné, ale praktické)
        actual_result_right = scalar * m
        np_actual_right = matrix_to_numpy(actual_result_right)
        np.testing.assert_array_equal(np_actual_right, expected_result)

    def test_matrix_multiplication(self):
        """Test maticového súčinu.."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])

        np_m1 = matrix_to_numpy(m1)
        np_m2 = matrix_to_numpy(m2)

        expected_result = np.matmul(np_m1, np_m2)

        actual_result = m1 @ m2

        np_actual = matrix_to_numpy(actual_result)

        # Porovnanie výsledkov
        np.testing.assert_array_equal(np_actual, expected_result)

    def test_matrix_multiplication_different_shapes(self):
        """Test maticového súčinu na maticiach rôznych rozmerov."""
        m1 = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        m2 = Matrix([[7, 8], [9, 10], [11, 12]])  # 3x2

        np_m1 = matrix_to_numpy(m1)
        np_m2 = matrix_to_numpy(m2)

        expected_result = np.matmul(np_m1, np_m2)

        actual_result = m1 @ m2

        np_actual = matrix_to_numpy(actual_result)

        # Porovnanie výsledkov
        np.testing.assert_array_equal(np_actual, expected_result)

    def test_matrix_multiplication_different_invalid_shapes(self):
        """Test maticového súčinu na maticiach rôznych (neplatných) rozmerov."""
        m1 = Matrix([[7, 8, 9], [10, 11, 12]])  # 2x3
        m2 = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3

        # Maticový súčin 2x3 @ 2x3 nie je definovaný (potrebujeme aby m1.cols == m2.rows)
        with self.assertRaises(ValueError) as context:
            m1 @ m2

        # Mali by sme dostať chybovú hlášku (neplatný maticový súčin)
        self.assertIn("Cannot multiply", str(context.exception))


class TestMatrixInversion(unittest.TestCase):
    """Test operácie inverzie matice."""

    def test_invert_identity(self):
        """Testuje inverziu jednotkovej matice."""
        m = Matrix.identity(3)

        np_m = matrix_to_numpy(m)

        expected_result = np.linalg.inv(np_m)

        actual_result = m.invert()

        np_actual = matrix_to_numpy(actual_result)

        # Porovnanie výsledkov
        np.testing.assert_array_almost_equal(np_actual, expected_result)

    def test_invert_simple(self):
        """Test inverzie pomerne štandardnej (regulárnej) matice."""
        m = Matrix([[4, 7], [2, 6]])

        np_m = matrix_to_numpy(m)

        expected_result = np.linalg.inv(np_m)

        actual_result = m.invert()

        np_actual = matrix_to_numpy(actual_result)

        # Porovnanie výsledkov
        np.testing.assert_array_almost_equal(np_actual, expected_result)

    def test_invert_singular(self):
        """Test inverzie singulárnej matice."""
        m = Matrix([[1, 2], [2, 4]])

        # Mali by sme dostať chybu (singulárna matica nemá inverznú maticu)
        with self.assertRaises(ValueError):
            m.invert()


class TestMatrixRank(unittest.TestCase):
    """Test vlastnosť hodnosti matice."""

    def test_rank_full_rank(self):
        """Testuje hodnosť matice, kde n = rank"""
        m = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        np_m = matrix_to_numpy(m)

        expected_result = np.linalg.matrix_rank(np_m)

        actual_result = m.rank()

        self.assertEqual(actual_result, expected_result)

    def test_rank_deficient(self):
        """Testuje hodnosť matice, kde hodnosť je menšia ako n"""
        m = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])

        np_m = matrix_to_numpy(m)

        expected_result = np.linalg.matrix_rank(np_m)

        actual_result = m.rank()

        self.assertEqual(actual_result, expected_result)

    def test_rank_empty(self):
        """Testuje hodnosť prázdnej matice."""
        m = Matrix([])

        actual_result = m.rank()

        self.assertEqual(actual_result, 0)


class TestMatrixRegularity(unittest.TestCase):
    """Testuje vlastnosť regulárnosti matice."""

    def test_is_regular_true(self):
        """Test regulárnosti na regulárnej matici."""
        m = Matrix([[1, 2], [3, 4]])

        np_m = matrix_to_numpy(m)

        expected_result = np.linalg.det(np_m) != 0

        actual_result = m.is_regular

        self.assertEqual(actual_result, expected_result)

    def test_is_regular_false(self):
        """Test regulárnosti na singulárnej matici."""
        m = Matrix([[1, 2], [2, 4]])

        np_m = matrix_to_numpy(m)

        expected_result = np.linalg.det(np_m) != 0

        actual_result = m.is_regular

        self.assertEqual(actual_result, expected_result)

    def test_is_regular_non_square(self):
        """Test regulárnosti matice, ktorá nie je štvorcová (= nemôže byť regulárna)."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        self.assertFalse(m.is_regular)


class TestEdgeCases(unittest.TestCase):
    """Testuje okrajové prípady (edge cases)."""

    def test_single_element_matrix(self):
        """Čo keď je matica rozmerov 1x1?."""
        m = Matrix([[5]])
        m.rref()
        self.assertEqual(m[1, 1], 1)

    def test_empty_matrix_operations(self):
        """Čo keď je matica prázdna?."""
        m = Matrix([])
        m.ref()
        m.rref()
        m.transpose()
        self.assertTrue(m.empty)

    def test_matrix_with_large_numbers(self):
        """Čo keď sú v matici pomerne veľké čísla?"""
        m = Matrix([[1000000, 2000000], [3000000, 4000000]])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)

    def test_matrix_with_small_numbers(self):
        """Čo keď sú v matici pomerne malé čísla?."""
        m = Matrix([[0.0001, 0.0002], [0.0003, 0.0004]])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)


class TestMatrixSolutions(unittest.TestCase):
    """Test riešení homogénnych a nehomogénnych sústav."""

    def _verify_solution(self, A_data, num_free_vars_expected=None, should_have_solution=True):
        """
        Helper funkcia na overenie riešenia porovnaním výstupu s NumPy.
        A_data: rozšírená matica sústavy (A|b)
        """
        m = Matrix(A_data)
        result = m.get_solutions()

        A_np = np.array([row[:-1] for row in A_data], dtype=float)
        b_np = np.array([row[-1] for row in A_data], dtype=float).reshape(-1, 1)

        rank_A = np.linalg.matrix_rank(A_np)
        rank_Ab = np.linalg.matrix_rank(np.column_stack([A_np, b_np]))

        # Test riešiteľnosti (Frobeniova veta)
        if rank_A != rank_Ab:
            self.assertEqual(result, "No solution.")
            self.assertFalse(should_have_solution)
            return

        self.assertTrue(should_have_solution)

        # Spočítaj počet voľných premenných
        if A_np.size > 0:
            n = A_np.shape[1]
            num_free = n - rank_A
        else:
            num_free = len(A_data[0]) - 1  # prázdna matica

        if num_free_vars_expected is not None:
            self.assertEqual(num_free, num_free_vars_expected,
                           f"Očakávaný počet voľných premenných: {num_free_vars_expected}, skutočný počet: {num_free}")

        # Kontrola parametrov v riešení
        for i in range(num_free):
            self.assertIn(f"t{i}", result, f"Chýba parameter t{i} pre danú voľnú premennú")

    # === NEHOMOGÉNNE SÚSTAVY ===

    def test_no_solution_inconsistent(self):
        """Degenerovaná sústava - nemá riešenie."""
        # x + y = 2
        # x + y = 5  (spor)
        self._verify_solution([[1, 1, 2], [1, 1, 5]], should_have_solution=False)

    def test_no_solution_rank_defect(self):
        """Rozšírená matica má vyššiu hodnosť ako nerozšírená - nemá riešenie."""
        # x + 2y + 3z = 1
        # 2x + 4y + 6z = 3  (= 2 * prvý riadok by mala byť na PS - 2, nie 3)
        self._verify_solution([[1, 2, 3, 1], [2, 4, 6, 3]], should_have_solution=False)

    def test_unique_solution_2x2(self):
        """Jedno riešenie matice 2x2."""
        # 2x + y = 5
        # x - y = 1
        # Riešenie: x=2, y=1
        self._verify_solution([[2, 1, 5], [1, -1, 1]], num_free_vars_expected=0)

    def test_unique_solution_3x3(self):
        """Jedno riešenie matice 3x3."""
        # x + y + z = 6
        # 2x - y + z = 3
        # x + 2y - z = 2
        self._verify_solution([[1, 1, 1, 6], [2, -1, 1, 3], [1, 2, -1, 2]], num_free_vars_expected=0)

    def test_infinite_solutions_underdetermined(self):
        """Nekonečne veľa riešení (1 rovnica, 2 neznáme)."""
        # x + 2y = 5
        # Voľná: y (t0), Bázická: x = 5 - 2t0
        self._verify_solution([[1, 2, 5]], num_free_vars_expected=1)

    def test_infinite_solutions_2eq_3vars(self):
        """2 rovnice, 3 neznáme - 1 voľná premenná."""
        # x + y + z = 1
        # 2x + y - z = 0
        self._verify_solution([[1, 1, 1, 1], [2, 1, -1, 0]], num_free_vars_expected=1)

    def test_infinite_solutions_rank1_3vars(self):
        """Hodnosť 1, 3 premenné - 2 voľné premenné."""
        # x + 2y + 3z = 4
        # 2x + 4y + 6z = 8  (násobok prvého)
        self._verify_solution([[1, 2, 3, 4], [2, 4, 6, 8]], num_free_vars_expected=2)

    def test_alternating_pivot_free_vars(self):
        """Pivoty a voľné premenné sa striedajú."""
        # x1 + 2x2 + 0x3 - x4 = 5
        # 0x1 + 0x2 + x3 + 3x4 = 2
        # Pivoty v stĺpcoch 1,3; voľné v stĺpcoch 2,4
        self._verify_solution([[1, 2, 0, -1, 5], [0, 0, 1, 3, 2], [0, 0, 0, 0, 0]],
                            num_free_vars_expected=2)

    def test_complex_non_homogenous(self):
        """Pomerne komplexná nehomogénna sústava."""
        data = [[6, -4, 9, 8, 6, 7],
                [4, -1, 6, 2, -1, 8],
                [6, 2, 4, -3, -15, 9]]
        self._verify_solution(data)

    # === HOMOGÉNNE SÚSTAVY ===

    def test_homogenous_unique_solution(self):
        """Homogénna sústava s jedným (triviálnym) riešením."""
        # x + y = 0
        # x - y = 0
        # Riešenie: x=0, y=0
        self._verify_solution([[1, 1, 0], [1, -1, 0]], num_free_vars_expected=0)

    def test_homogenous_infinite_solutions_1free(self):
        """Homogénna sústava s nekonečne veľa riešeniami - 1 voľná premenná."""
        # x + y + z = 0
        # 2x + 2y + 2z = 0
        self._verify_solution([[1, 1, 1, 0], [2, 2, 2, 0]], num_free_vars_expected=2)

    def test_homogenous_infinite_solutions_2free(self):
        """Homogénna sústava s nekonečne veľa riešeniami - 2 voľné premenné."""
        # x + y + z = 0
        self._verify_solution([[1, 1, 1, 0]], num_free_vars_expected=2)

    def test_homogenous_3x4_rank2(self):
        """Homogénna sústava 3x4, hodnosť 2 - 2 voľné premenné."""
        # x + 2y + 3z + 4w = 0
        # 2x + 4y + 5z + 6w = 0
        # 3x + 6y + 8z + 10w = 0
        self._verify_solution([[1, 2, 3, 4, 0], [2, 4, 5, 6, 0], [3, 6, 8, 10, 0]],
                            num_free_vars_expected=2)

    # === OKRAJOVÉ PRÍPADY ===

    def test_zero_matrix_homogenous(self):
        """Nulová matica - každé riešenie je riešením (celé univerzum)."""
        m = Matrix([[0, 0, 0], [0, 0, 0]])
        result = m.get_solutions()
        self.assertIn("domain", result.lower())

    def test_zero_matrix_all_free_3vars(self):
        """Nulová matica 2x4 - všetky 3/4 (nehomogénna/homogénna) premenné voľné."""
        # 0x + 0y + 0z = 0
        # 0x + 0y + 0z = 0
        # Po RREF to bude prázdna matica
        m = Matrix([[0, 0, 0, 0], [0, 0, 0, 0]])
        result = m.get_solutions()
        self.assertIn("domain", result.lower())

    def test_single_variable_system(self):
        """Nehomogénna sústava s 1 premennou."""
        # 2x = 4 => x = 2
        self._verify_solution([[2, 4]], num_free_vars_expected=0)

    def test_single_variable_homogenous(self):
        """Homogénna sústava s 1 premennou."""
        # 3x = 0 => x = 0
        self._verify_solution([[3, 0]], num_free_vars_expected=0)

    def test_more_rows_than_cols_valid(self):
        """Sústava rovníc, kde je viac rovníc ako premenných a má riešenie."""
        # x + y = 3
        # 2x + 2y = 6
        # 3x + 3y = 9
        self._verify_solution([[1, 1, 3], [2, 2, 6], [3, 3, 9]], num_free_vars_expected=1)

    def test_more_rows_than_cols_invalid(self):
        """Sústava rovníc, kde je viac rovníc ako premenných a nemá riešenie."""
        # x + y = 3
        # 2x + 2y = 6
        # x + y = 5  (spor s prvou rovnicou)
        self._verify_solution([[1, 1, 3], [2, 2, 6], [1, 1, 5]], should_have_solution=False)

    def test_identity_matrix_system(self):
        """Jednotková matica - má iba jedno riešenie."""
        # x = 1
        # y = 2
        # z = 3
        self._verify_solution([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]], num_free_vars_expected=0)

    def test_all_pivots_at_end(self):
        """Všetky (jeden) pivoty na sú konci stĺpcov."""
        # 0x + 0y + z = 1
        # x a y sú voľné, z je bázická
        self._verify_solution([[0, 0, 1, 1]], num_free_vars_expected=2)

    def test_fractional_coefficients(self):
        """Sústava so zlomkovými koeficientami."""
        # 0.5x + 0.25y = 1
        # 1.5x + 0.75y = 3
        self._verify_solution([[0.5, 0.25, 1], [1.5, 0.75, 3]], num_free_vars_expected=1)

    def test_negative_solution(self):
        """Sústava so zápornými riešeniami."""
        # x + 2y = -3
        # 3x - y = -5
        self._verify_solution([[1, 2, -3], [3, -1, -5]], num_free_vars_expected=0)

    def test_large_system_5x6(self):
        """Väčšia sústava 5x6."""
        data = [[1, 2, 3, 4, 5, 1],
                [2, 4, 6, 8, 10, 2],
                [1, 1, 1, 1, 1, 0],
                [0, 1, 2, 3, 4, 1],
                [1, 0, 0, 0, 1, 2]]
        self._verify_solution(data)

    def test_zero_row_in_middle(self):
        """Nulový riadok uprostred matice."""
        # x + y = 1
        # 0 = 0
        # 2x + 2y = 2
        self._verify_solution([[1, 1, 1], [0, 0, 0], [2, 2, 2]], num_free_vars_expected=1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
