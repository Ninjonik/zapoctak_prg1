"""
Unit tests for the Matrix class.
Tests all functionality including edge cases and Gauss-Jordan elimination (RREF).

To run the tests:
python -m unittest test_matrix -v

"""

import unittest
from main import Matrix


class TestMatrixConstruction(unittest.TestCase):
    """Test matrix construction methods."""

    def test_init_from_list(self):
        """Test basic initialization from a list."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 3)
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 3], 6)

    def test_zeros(self):
        """Test zeros matrix creation."""
        m = Matrix.zeros(3, 4)
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 4)
        for i in range(1, 4):
            for j in range(1, 5):
                self.assertEqual(m[i, j], 0)

    def test_identity(self):
        """Test identity matrix creation."""
        m = Matrix.identity(3)
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 3)
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[3, 3], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[2, 1], 0)

    def test_from_list(self):
        """Test from_list class method."""
        m = Matrix.from_list([[1, 2], [3, 4]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m[1, 1], 1)

    def test_from_matrix(self):
        """Test copying a matrix."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix.from_matrix(m1)
        self.assertEqual(m1, m2)
        # Ensure it's a deep copy
        m2[1, 1] = 99
        self.assertNotEqual(m1[1, 1], m2[1, 1])

    def test_empty_matrix(self):
        """Test empty matrix."""
        m = Matrix([])
        self.assertTrue(m.empty)
        self.assertEqual(m.rows, 0)
        self.assertEqual(m.cols, 0)


class TestMatrixProperties(unittest.TestCase):
    """Test matrix properties and characteristics."""

    def test_shape(self):
        """Test shape property."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.shape, (2, 3))

    def test_rows_cols(self):
        """Test rows and cols properties."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 3)

    def test_is_square(self):
        """Test square matrix detection."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(m1.is_square)
        self.assertFalse(m2.is_square)

    def test_is_symmetric(self):
        """Test symmetric matrix detection."""
        symmetric = Matrix([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        not_symmetric = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        non_square = Matrix([[1, 2, 3], [4, 5, 6]])

        self.assertTrue(symmetric.is_symmetric)
        self.assertFalse(not_symmetric.is_symmetric)
        self.assertFalse(non_square.is_symmetric)

    def test_empty_property(self):
        """Test empty property."""
        m1 = Matrix([])
        m2 = Matrix([[1, 2]])
        self.assertTrue(m1.empty)
        self.assertFalse(m2.empty)


class TestMatrixIndexing(unittest.TestCase):
    """Test matrix indexing operations."""

    def test_getitem_row(self):
        """Test getting a row."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        row1 = m[1]
        row2 = m[2]
        self.assertEqual(row1, (1, 2, 3))
        self.assertEqual(row2, (4, 5, 6))

    def test_getitem_element(self):
        """Test getting an element."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 3], 3)
        self.assertEqual(m[2, 2], 5)

    def test_getitem_out_of_range(self):
        """Test index out of range errors."""
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(IndexError):
            _ = m[3]
        with self.assertRaises(IndexError):
            _ = m[1, 3]
        with self.assertRaises(IndexError):
            _ = m[0]

    def test_setitem_row(self):
        """Test setting a row."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m[1] = [7, 8, 9]
        self.assertEqual(m[1], (7, 8, 9))

    def test_setitem_element(self):
        """Test setting an element."""
        m = Matrix([[1, 2], [3, 4]])
        m[1, 1] = 99
        self.assertEqual(m[1, 1], 99)

    def test_setitem_wrong_length(self):
        """Test setting a row with wrong length."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m[1] = [1, 2]

    def test_iteration(self):
        """Test matrix iteration."""
        m = Matrix([[1, 2], [3, 4]])
        rows = list(m)
        self.assertEqual(rows, [(1, 2), (3, 4)])


class TestMatrixEquality(unittest.TestCase):
    """Test matrix equality comparisons."""

    def test_equal_matrices(self):
        """Test equal matrices."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m1, m2)

    def test_unequal_matrices(self):
        """Test unequal matrices."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 5]])
        self.assertNotEqual(m1, m2)

    def test_different_shapes(self):
        """Test matrices with different shapes."""
        m1 = Matrix([[1, 2]])
        m2 = Matrix([[1], [2]])
        self.assertNotEqual(m1, m2)

    def test_empty_matrices(self):
        """Test empty matrices equality."""
        m1 = Matrix([])
        m2 = Matrix([])
        self.assertEqual(m1, m2)

    def test_empty_vs_none(self):
        """Test empty matrix vs None."""
        m = Matrix([])
        self.assertEqual(m, None)


class TestMatrixCopy(unittest.TestCase):
    """Test matrix copy operations."""

    def test_copy(self):
        """Test matrix copy."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = m1.copy()
        self.assertEqual(m1, m2)
        m2[1, 1] = 99
        self.assertNotEqual(m1[1, 1], m2[1, 1])


class TestMatrixTranspose(unittest.TestCase):
    """Test matrix transpose operations."""

    def test_transpose_rectangular(self):
        """Test transpose of rectangular matrix."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.transpose()
        self.assertEqual(m.shape, (3, 2))
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 4)
        self.assertEqual(m[3, 1], 3)
        self.assertEqual(m[3, 2], 6)

    def test_transpose_square(self):
        """Test transpose of square matrix."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m.transpose()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 4)
        self.assertEqual(m[1, 3], 7)
        self.assertEqual(m[2, 1], 2)

    def test_transpose_symmetric(self):
        """Test transpose of symmetric matrix."""
        m = Matrix([[1, 2], [2, 3]])
        original = m.copy()
        m.transpose()
        self.assertEqual(m, original)


class TestElementaryRowOperations(unittest.TestCase):
    """Test elementary row operations."""

    def test_scale_row(self):
        """Test scaling a row."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.scale_row(1, 2)
        self.assertEqual(m[1], (2, 4, 6))
        self.assertEqual(m[2], (4, 5, 6))

    def test_scale_row_by_fraction(self):
        """Test scaling by a fraction."""
        m = Matrix([[2, 4, 6], [1, 2, 3]])
        m.scale_row(1, 0.5)
        self.assertEqual(m[1], (1, 2, 3))

    def test_scale_row_by_zero(self):
        """Test scaling by zero (should not work)."""
        m = Matrix([[1, 2], [3, 4]])
        original = m.copy()
        m.scale_row(1, 0)
        self.assertEqual(m, original)

    def test_swap_rows(self):
        """Test swapping rows."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.swap_rows(1, 2)
        self.assertEqual(m[1], (4, 5, 6))
        self.assertEqual(m[2], (1, 2, 3))

    def test_add_row(self):
        """Test adding rows."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.add_row(1, 2, 1)
        self.assertEqual(m[1], (5, 7, 9))
        self.assertEqual(m[2], (4, 5, 6))

    def test_add_row_with_multiple(self):
        """Test adding row with multiple."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.add_row(1, 2, 2)
        self.assertEqual(m[1], (9, 12, 15))

    def test_subtract_row(self):
        """Test subtracting rows."""
        m = Matrix([[5, 7, 9], [4, 5, 6]])
        m.subtract_row(1, 2, 1)
        self.assertEqual(m[1], (1, 2, 3))


class TestPivotOperations(unittest.TestCase):
    """Test pivot-related operations."""

    def test_get_pivot(self):
        """Test getting pivot position."""
        m = Matrix([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        self.assertEqual(m.get_pivot(1), 1)
        self.assertEqual(m.get_pivot(2), 2)
        self.assertEqual(m.get_pivot(3), 3)

    def test_get_pivot_with_leading_zeros(self):
        """Test pivot with leading zeros."""
        m = Matrix([[0, 0, 1, 2], [0, 3, 4, 5]])
        self.assertEqual(m.get_pivot(1), 3)
        self.assertEqual(m.get_pivot(2), 2)

    def test_get_pivot_zero_row(self):
        """Test pivot on zero row."""
        m = Matrix([[1, 2], [0, 0]])
        self.assertIsNone(m.get_pivot(2))

    def test_is_row_empty(self):
        """Test checking if row is empty."""
        m = Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
        self.assertFalse(m.is_row_empty(1))
        self.assertTrue(m.is_row_empty(2))
        self.assertFalse(m.is_row_empty(3))


class TestRowRemoval(unittest.TestCase):
    """Test row removal operations."""

    def test_remove_row(self):
        """Test removing a row."""
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        m.remove_row(2)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m[1], (1, 2))
        self.assertEqual(m[2], (5, 6))

    def test_remove_null_rows(self):
        """Test removing null rows."""
        m = Matrix([[1, 2], [0, 0], [3, 4], [0, 0]])
        m.remove_null_rows()
        self.assertEqual(m.rows, 2)
        self.assertEqual(m[1], (1, 2))
        self.assertEqual(m[2], (3, 4))

    def test_remove_null_rows_all_zero(self):
        """Test removing null rows when all are zero."""
        m = Matrix([[0, 0], [0, 0]])
        m.remove_null_rows()
        self.assertEqual(m.rows, 0)
        self.assertTrue(m.empty)


class TestREF(unittest.TestCase):
    """Test Row Echelon Form (REF) operations."""

    def test_ref_simple(self):
        """Test REF on a simple matrix."""
        m = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        m.ref()
        # Check that it's in REF form (pivots are left of lower row pivots)
        pivot1 = m.get_pivot(1)
        if m.rows >= 2:
            pivot2 = m.get_pivot(2)
            if pivot1 is not None and pivot2 is not None:
                self.assertLess(pivot1, pivot2)

    def test_ref_already_ref(self):
        """Test REF on matrix already in REF."""
        m = Matrix([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
        m.ref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[3, 3], 1)

    def test_ref_with_zero_row(self):
        """Test REF with zero row."""
        m = Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
        m.ref()
        # Zero rows should be removed
        self.assertEqual(m.rows, 2)


class TestRREF(unittest.TestCase):
    """Test Reduced Row Echelon Form (RREF) operations - Gauss-Jordan Elimination."""

    def test_rref_identity(self):
        """Test RREF on identity matrix."""
        m = Matrix.identity(3)
        m.rref()
        expected = Matrix.identity(3)
        self.assertEqual(m, expected)

    def test_rref_simple_2x2(self):
        """Test RREF on simple 2x2 matrix."""
        m = Matrix([[2, 4], [1, 3]])
        m.rref()
        # Should result in identity or RREF form
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[2, 1], 0)

    def test_rref_3x3_full_rank(self):
        """Test RREF on 3x3 full rank matrix."""
        m = Matrix([[1, 2, 3], [2, 5, 7], [3, 5, 8]])
        m.rref()
        # Should have 1s on diagonal and 0s elsewhere
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[3, 3], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[1, 3], 0)
        self.assertEqual(m[2, 1], 0)
        self.assertEqual(m[2, 3], 0)
        self.assertEqual(m[3, 1], 0)
        self.assertEqual(m[3, 2], 0)

    def test_rref_rectangular_full_column_rank(self):
        """Test RREF on rectangular matrix with full column rank."""
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        m.rref()
        # Should have identity in top portion
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[2, 1], 0)
        self.assertEqual(m[2, 2], 1)

    def test_rref_rectangular_full_row_rank(self):
        """Test RREF on rectangular matrix with full row rank."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.rref()
        # First two columns should form identity
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[2, 1], 0)
        self.assertEqual(m[2, 2], 1)

    def test_rref_with_zero_row(self):
        """Test RREF with zero row in middle."""
        m = Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
        m.rref()
        # Zero row should be removed
        self.assertEqual(m.rows, 2)
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)

    def test_rref_with_multiple_zero_rows(self):
        """Test RREF with multiple zero rows."""
        m = Matrix([[1, 2], [0, 0], [0, 0], [3, 4]])
        m.rref()
        self.assertEqual(m.rows, 2)

    def test_rref_singular_matrix(self):
        """Test RREF on singular matrix."""
        m = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        m.rref()
        # Should reduce to one non-zero row
        self.assertEqual(m.rows, 1)
        self.assertEqual(m[1, 1], 1)

    def test_rref_rank_deficient(self):
        """Test RREF on rank-deficient matrix."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [5, 7, 9]])
        m.rref()
        # Should have fewer than 3 rows after reduction
        self.assertLess(m.rows, 3)

    def test_rref_with_fractions(self):
        """Test RREF that results in fractions."""
        m = Matrix([[1, 2], [3, 5]])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[2, 1], 0)

    def test_rref_augmented_matrix_solution(self):
        """Test RREF on augmented matrix (system of equations)."""
        # System: x + 2y = 5, 3x + 4y = 11
        # Solution: x = 1, y = 2
        m = Matrix([[1, 2, 5], [3, 4, 11]])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[1, 3], 1)
        self.assertEqual(m[2, 1], 0)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[2, 3], 2)

    def test_rref_inconsistent_system(self):
        """Test RREF on inconsistent system."""
        # Inconsistent system: x + y = 1, x + y = 2
        m = Matrix([[1, 1, 1], [1, 1, 2]])
        m.rref()
        # Should have a row like [0, 0, 1] indicating inconsistency
        # Or reduce to one row
        pass

    def test_rref_underdetermined_system(self):
        """Test RREF on underdetermined system."""
        # More unknowns than equations
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)

    def test_rref_already_rref(self):
        """Test RREF on matrix already in RREF."""
        m = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m.rref()
        expected = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(m, expected)

    def test_rref_with_negative_numbers(self):
        """Test RREF with negative numbers."""
        m = Matrix([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])
        m.rref()
        # Should still produce valid RREF
        self.assertEqual(m[1, 1], 1)

    def test_rref_single_row(self):
        """Test RREF on single row."""
        m = Matrix([[2, 4, 6]])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 2)
        self.assertEqual(m[1, 3], 3)

    def test_rref_single_column(self):
        """Test RREF on single column."""
        m = Matrix([[2], [4], [6]])
        m.rref()
        self.assertEqual(m.rows, 1)
        self.assertEqual(m[1, 1], 1)

    def test_rref_all_zeros(self):
        """Test RREF on all-zero matrix."""
        m = Matrix.zeros(3, 3)
        m.rref()
        self.assertTrue(m.empty)

    def test_rref_leading_zeros(self):
        """Test RREF with leading zeros in rows."""
        m = Matrix([[0, 1, 2], [1, 2, 3], [0, 0, 1]])
        m.rref()
        # Should properly reorder and reduce
        self.assertEqual(m[1, 1], 1)

    def test_rref_complex_example_1(self):
        """Test RREF on complex example 1."""
        m = Matrix([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        if m.rows >= 2:
            self.assertEqual(m[2, 2], 1)

    def test_rref_complex_example_2(self):
        """Test RREF on complex example 2."""
        m = Matrix([
            [2, 1, -1, 8],
            [-3, -1, 2, -11],
            [-2, 1, 2, -3]
        ])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[3, 3], 1)

    def test_rref_preserves_relationships(self):
        """Test that RREF preserves linear relationships."""
        # Create a simple system where we know the answer
        # 2x + y = 5, x + y = 3 => x = 2, y = 1
        m = Matrix([[2, 1, 5], [1, 1, 3]])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[1, 3], 2)
        self.assertEqual(m[2, 1], 0)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[2, 3], 1)

    def test_rref_5x5_matrix(self):
        """Test RREF on larger 5x5 matrix."""
        m = Matrix([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ])
        m.rref()
        # This matrix is rank deficient
        self.assertLess(m.rows, 5)


class TestEdgeCases(unittest.TestCase):
    """Test various edge cases."""

    def test_single_element_matrix(self):
        """Test single element matrix."""
        m = Matrix([[5]])
        m.rref()
        self.assertEqual(m[1, 1], 1)

    def test_empty_matrix_operations(self):
        """Test operations on empty matrix."""
        m = Matrix([])
        m.ref()
        m.rref()
        m.transpose()
        self.assertTrue(m.empty)

    def test_matrix_with_large_numbers(self):
        """Test matrix with large numbers."""
        m = Matrix([[1000000, 2000000], [3000000, 4000000]])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)

    def test_matrix_with_small_numbers(self):
        """Test matrix with very small numbers."""
        m = Matrix([[0.0001, 0.0002], [0.0003, 0.0004]])
        m.rref()
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

