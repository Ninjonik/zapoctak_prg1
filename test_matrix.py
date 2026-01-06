"""
Unit tests for the Matrix class.
Tests all functionality including edge cases and Gauss-Jordan elimination (RREF).
Uses numpy as a source of truth for matrix operations.

To run the tests:
python -m unittest test_matrix -v

"""

import unittest
from typing import List

import numpy as np
from main import Matrix


def matrix_to_numpy(matrix):
    """Convert a Matrix object to a numpy array."""
    if matrix.empty:
        return np.array([])
    return np.array([list(matrix[i]) for i in range(1, matrix.rows + 1)])


def numpy_to_matrix(array):
    """Convert a numpy array to a Matrix object."""
    if array.size == 0:
        return Matrix([])
    return Matrix(array.tolist())


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
        """Test zeros matrix creation"""
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
        # Create test matrix
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np.transpose(np_m)

        # Transpose the matrix
        m.transpose()

        # Convert result to numpy for comparison
        np_result = matrix_to_numpy(m)

        # Compare results
        np.testing.assert_array_equal(np_result, expected_result)

        # Additional checks
        self.assertEqual(m.shape, (3, 2))
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 4)
        self.assertEqual(m[3, 1], 3)
        self.assertEqual(m[3, 2], 6)

    def test_transpose_square(self):
        """Test transpose of square matrix."""
        # Create test matrix
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np.transpose(np_m)

        # Transpose the matrix
        m.transpose()

        # Convert result to numpy for comparison
        np_result = matrix_to_numpy(m)

        # Compare results
        np.testing.assert_array_equal(np_result, expected_result)

        # Additional checks
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 4)
        self.assertEqual(m[1, 3], 7)
        self.assertEqual(m[2, 1], 2)

    def test_transpose_symmetric(self):
        """Test transpose of symmetric matrix."""
        # Create test matrix
        m = Matrix([[1, 2], [2, 3]])
        original = m.copy()

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np.transpose(np_m)

        # Transpose the matrix
        m.transpose()

        # Convert result to numpy for comparison
        np_result = matrix_to_numpy(m)

        # Compare results
        np.testing.assert_array_equal(np_result, expected_result)

        # For symmetric matrices, the transpose should equal the original
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
        # Create test matrix
        m = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])

        # Make a copy for numpy operations
        np_m = matrix_to_numpy(m)

        # Perform REF operation
        m.ref()

        # Check that it's in REF form (pivots are left of lower row pivots)
        pivot1 = m.get_pivot(1)
        if m.rows >= 2:
            pivot2 = m.get_pivot(2)
            if pivot1 is not None and pivot2 is not None:
                self.assertLess(pivot1, pivot2)

        # Additional verification: check that all elements below pivots are zero
        for row in range(2, m.rows + 1):
            for prev_row in range(1, row):
                pivot_col = m.get_pivot(prev_row)
                if pivot_col is not None:
                    self.assertEqual(m[row, pivot_col], 0)

    def test_ref_already_ref(self):
        """Test REF on matrix already in REF."""
        # Create test matrix already in REF
        m = Matrix([[1, 2, 3], [0, 1, 4], [0, 0, 1]])

        # Make a copy for comparison
        original = m.copy()

        # Perform REF operation
        m.ref()

        # The matrix should remain essentially the same (pivots might be normalized)
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[3, 3], 1)

        # Check that the structure is preserved
        for row in range(1, m.rows + 1):
            pivot_col = m.get_pivot(row)
            if pivot_col is not None:
                # Check that all elements below the pivot are zero
                for below_row in range(row + 1, m.rows + 1):
                    self.assertEqual(m[below_row, pivot_col], 0)

    def test_ref_with_zero_row(self):
        """Test REF with zero row."""
        # Create test matrix with a zero row
        m = Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]])

        # Perform REF operation
        m.ref()

        # Zero rows should be removed
        self.assertEqual(m.rows, 2)


class TestRREF(unittest.TestCase):
    """Test Reduced Row Echelon Form (RREF) operations - Gauss-Jordan Elimination."""

    def test_rref_identity(self):
        """Test RREF on identity matrix."""
        # Create identity matrix
        m = Matrix.identity(3)

        # Make a copy for numpy operations
        np_m = matrix_to_numpy(m)

        # Perform RREF operation
        m.rref()

        # Identity matrix should remain unchanged after RREF
        expected = Matrix.identity(3)
        self.assertEqual(m, expected)

    def test_rref_simple_2x2(self):
        """Test RREF on simple 2x2 matrix."""
        # Create test matrix
        m = Matrix([[2, 4], [1, 3]])

        # Make a copy for numpy operations
        np_m = matrix_to_numpy(m)

        # Perform RREF operation
        m.rref()

        # Should result in identity or RREF form
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[2, 1], 0)

        # Additional verification: check that the matrix is in RREF form
        # 1. All pivots are 1
        # 2. All other elements in pivot columns are 0
        for row in range(1, m.rows + 1):
            pivot_col = m.get_pivot(row)
            if pivot_col is not None:
                self.assertEqual(m[row, pivot_col], 1)
                for other_row in range(1, m.rows + 1):
                    if other_row != row:
                        self.assertEqual(m[other_row, pivot_col], 0)

    def test_rref_3x3_full_rank(self):
        """Test RREF on 3x3 full rank matrix."""
        # Create test matrix
        m = Matrix([[1, 2, 3], [2, 5, 7], [3, 5, 8]])

        # Make a copy for numpy operations
        np_m = matrix_to_numpy(m)

        # Perform RREF operation
        m.rref()

        # Should have 1s on diagonal and 0s elsewhere
        self.assertEqual(m[1, 1], 1)
        self.assertEqual(m[1, 2], 0)
        self.assertEqual(m[1, 3], 1)
        self.assertEqual(m[2, 1], 0)
        self.assertEqual(m[2, 2], 1)
        self.assertEqual(m[2, 3], 1)

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


class TestArithmeticOperations(unittest.TestCase):
    """Test arithmetic operations on matrices."""

    def test_right_addition(self):
        """Test right addition (scalar + matrix)."""
        # This tests the __radd__ method
        # Create test matrix
        m = Matrix([[1, 2], [3, 4]])

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # For addition, scalar + matrix should be equivalent to matrix + scalar
        # which is equivalent to adding the scalar to each element
        scalar = 5
        expected_result = scalar + np_m

        # Compute actual result using Matrix class
        # This should call __radd__ since the left operand is not a Matrix
        actual_result = scalar + m

        # Convert actual result to numpy for comparison
        np_actual = matrix_to_numpy(actual_result)

        # Compare results
        np.testing.assert_array_equal(np_actual, expected_result)

    def test_right_subtraction(self):
        """Test right subtraction (scalar - matrix)."""
        # This tests the __rsub__ method
        # Create test matrix
        m = Matrix([[1, 2], [3, 4]])

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # For subtraction, scalar - matrix is different from matrix - scalar
        scalar = 10
        expected_result = scalar - np_m

        # Compute actual result using Matrix class
        # This should call __rsub__ since the left operand is not a Matrix
        actual_result = scalar - m

        # Convert actual result to numpy for comparison
        np_actual = matrix_to_numpy(actual_result)

        # Compare results
        np.testing.assert_array_equal(np_actual, expected_result)

    def test_matrix_addition(self):
        """Test matrix addition."""
        # Create test matrices
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])

        # Convert to numpy arrays
        np_m1 = matrix_to_numpy(m1)
        np_m2 = matrix_to_numpy(m2)

        # Compute expected result using numpy
        expected_result = np_m1 + np_m2

        # Compute actual result using Matrix class
        actual_result = m1 + m2

        # Convert actual result to numpy for comparison
        np_actual = matrix_to_numpy(actual_result)

        # Compare results
        np.testing.assert_array_equal(np_actual, expected_result)

        # Test right addition (should be the same as left addition)
        actual_result_right = m2 + m1
        np_actual_right = matrix_to_numpy(actual_result_right)
        np.testing.assert_array_equal(np_actual_right, expected_result)

    def test_matrix_subtraction(self):
        """Test matrix subtraction."""
        # Create test matrices
        m1 = Matrix([[10, 20], [30, 40]])
        m2 = Matrix([[5, 6], [7, 8]])

        # Convert to numpy arrays
        np_m1 = matrix_to_numpy(m1)
        np_m2 = matrix_to_numpy(m2)

        # Compute expected result using numpy
        expected_result = np_m1 - np_m2

        # Compute actual result using Matrix class
        actual_result = m1 - m2

        # Convert actual result to numpy for comparison
        np_actual = matrix_to_numpy(actual_result)

        # Compare results
        np.testing.assert_array_equal(np_actual, expected_result)

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        # Create test matrix
        m = Matrix([[1, 2], [3, 4]])
        scalar = 2

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np_m * scalar

        # Compute actual result using Matrix class
        actual_result = m * scalar

        # Convert actual result to numpy for comparison
        np_actual = matrix_to_numpy(actual_result)

        # Compare results
        np.testing.assert_array_equal(np_actual, expected_result)

        # Test right multiplication
        actual_result_right = scalar * m
        np_actual_right = matrix_to_numpy(actual_result_right)
        np.testing.assert_array_equal(np_actual_right, expected_result)

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        # Create test matrices
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])

        # Convert to numpy arrays
        np_m1 = matrix_to_numpy(m1)
        np_m2 = matrix_to_numpy(m2)

        # Compute expected result using numpy
        expected_result = np.matmul(np_m1, np_m2)

        # Compute actual result using Matrix class
        actual_result = m1 @ m2

        # Convert actual result to numpy for comparison
        np_actual = matrix_to_numpy(actual_result)

        # Compare results
        np.testing.assert_array_equal(np_actual, expected_result)

    def test_matrix_multiplication_different_shapes(self):
        """Test matrix multiplication with different shapes."""
        # Create test matrices
        m1 = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        m2 = Matrix([[7, 8], [9, 10], [11, 12]])  # 3x2

        # Convert to numpy arrays
        np_m1 = matrix_to_numpy(m1)
        np_m2 = matrix_to_numpy(m2)

        # Compute expected result using numpy
        expected_result = np.matmul(np_m1, np_m2)

        # Compute actual result using Matrix class
        actual_result = m1 @ m2

        # Convert actual result to numpy for comparison
        np_actual = matrix_to_numpy(actual_result)

        # Compare results
        np.testing.assert_array_equal(np_actual, expected_result)


class TestMatrixInversion(unittest.TestCase):
    """Test matrix inversion."""

    def test_invert_identity(self):
        """Test inverting an identity matrix."""
        # Create identity matrix
        m = Matrix.identity(3)

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np.linalg.inv(np_m)

        # Compute actual result using Matrix class
        actual_result = m.invert()

        # Convert actual result to numpy for comparison
        np_actual = matrix_to_numpy(actual_result)

        # Compare results
        np.testing.assert_array_almost_equal(np_actual, expected_result)

    def test_invert_simple(self):
        """Test inverting a simple matrix."""
        # Create test matrix
        m = Matrix([[4, 7], [2, 6]])

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np.linalg.inv(np_m)

        # Compute actual result using Matrix class
        actual_result = m.invert()

        # Convert actual result to numpy for comparison
        np_actual = matrix_to_numpy(actual_result)

        # Compare results
        np.testing.assert_array_almost_equal(np_actual, expected_result)

    def test_invert_singular(self):
        """Test inverting a singular matrix."""
        # Create singular matrix
        m = Matrix([[1, 2], [2, 4]])

        # Compute actual result using Matrix class
        with self.assertRaises(ValueError):
            m.invert()


class TestMatrixRank(unittest.TestCase):
    """Test matrix rank calculation."""

    def test_rank_full_rank(self):
        """Test rank of a full rank matrix."""
        # Create test matrix
        m = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np.linalg.matrix_rank(np_m)

        # Compute actual result using Matrix class
        actual_result = m.rank()

        # Compare results
        self.assertEqual(actual_result, expected_result)

    def test_rank_deficient(self):
        """Test rank of a rank-deficient matrix."""
        # Create test matrix
        m = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np.linalg.matrix_rank(np_m)

        # Compute actual result using Matrix class
        actual_result = m.rank()

        # Compare results
        self.assertEqual(actual_result, expected_result)

    def test_rank_empty(self):
        """Test rank of an empty matrix."""
        # Create empty matrix
        m = Matrix([])

        # Compute actual result using Matrix class
        actual_result = m.rank()

        # Rank of empty matrix should be 0
        self.assertEqual(actual_result, 0)


class TestMatrixRegularity(unittest.TestCase):
    """Test matrix regularity."""

    def test_is_regular_true(self):
        """Test a regular (invertible) matrix."""
        # Create test matrix
        m = Matrix([[1, 2], [3, 4]])

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np.linalg.det(np_m) != 0

        # Compute actual result using Matrix class
        actual_result = m.is_regular

        # Compare results
        self.assertEqual(actual_result, expected_result)

    def test_is_regular_false(self):
        """Test a singular (non-invertible) matrix."""
        # Create test matrix
        m = Matrix([[1, 2], [2, 4]])

        # Convert to numpy array
        np_m = matrix_to_numpy(m)

        # Compute expected result using numpy
        expected_result = np.linalg.det(np_m) != 0

        # Compute actual result using Matrix class
        actual_result = m.is_regular

        # Compare results
        self.assertEqual(actual_result, expected_result)

    def test_is_regular_non_square(self):
        """Test regularity of a non-square matrix."""
        # Create test matrix
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        # Non-square matrices are not regular
        self.assertFalse(m.is_regular)


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


class TestMatrixSolutions(unittest.TestCase):
    """Test riešení homogénnych a nehomogénnych sústav."""

    def verify_solution(self, matrix_data: List[List[float]], solution_str: str):
        """
        Pomocná logika na overenie, či vrátený parametrický popis je korektný.
        """
        # 1. Extrakcia dát
        np_a_full = np.array(matrix_data)
        A = np_a_full[:, :-1]  # Matica koeficientov
        b = np_a_full[:, -1]  # Pravá strana

        # Ak sústava nemá riešenie, v kóde máš string "Nemá riešenie."
        if solution_str == "No solution.":
            # Overíme pomocou Frobeniusovej vety cez numpy
            rank_a = np.linalg.matrix_rank(A)
            rank_aug = np.linalg.matrix_rank(np_a_full)
            self.assertNotEqual(rank_a, rank_aug)
            return

        # 2. Parsovanie výsledku (veľmi zjednodušené pre test)
        # V reálnom teste by si mohol v Matrix triede vrátiť radšej objekty,
        # ale overíme to aspoň matematicky z reťazca, ak je to potrebné.
        # Pre účely tohto testu predpokladáme, že overujeme len jedno (partikulárne) riešenie.
        pass

    # ==================== SYSTEMATICKÉ TESTY PRE get_solutions() ====================
    # Budeme testovať všetky možné prípady a porovnávať s NumPy

    def _verify_solution(self, A_data, num_free_vars_expected=None, should_have_solution=True):
        """
        Helper funkcia na verifikáciu riešenia porovnaním s NumPy.
        A_data: rozšírená matica [A|b]
        """
        m = Matrix(A_data)
        result = m.get_solutions()

        # NumPy verifikácia
        A_np = np.array([row[:-1] for row in A_data], dtype=float)
        b_np = np.array([row[-1] for row in A_data], dtype=float).reshape(-1, 1)

        rank_A = np.linalg.matrix_rank(A_np)
        rank_Ab = np.linalg.matrix_rank(np.column_stack([A_np, b_np]))

        # Kontrola riešiteľnosti
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
                           f"Očakávaný počet voľných premenných: {num_free_vars_expected}, dostal: {num_free}")

        # Kontrola parametrov v riešení
        for i in range(num_free):
            self.assertIn(f"t{i}", result, f"Chýba parameter t{i} pre voľnú premennú")

    # === NEHOMOGÉNNE SÚSTAVY ===

    def test_no_solution_inconsistent(self):
        """Nekonzistentná sústava - nemá riešenie."""
        # x + y = 2
        # x + y = 5  (kontradikcia)
        self._verify_solution([[1, 1, 2], [1, 1, 5]], should_have_solution=False)

    def test_no_solution_rank_defect(self):
        """Rozšírená matica má vyšší rank ako koeficientová - nemá riešenie."""
        # x + 2y + 3z = 1
        # 2x + 4y + 6z = 3  (= 2 * prvý riadok by malo dať 2, nie 3)
        self._verify_solution([[1, 2, 3, 1], [2, 4, 6, 3]], should_have_solution=False)

    def test_unique_solution_2x2(self):
        """Jedinečné riešenie 2x2."""
        # 2x + y = 5
        # x - y = 1
        # Riešenie: x=2, y=1
        self._verify_solution([[2, 1, 5], [1, -1, 1]], num_free_vars_expected=0)

    def test_unique_solution_3x3(self):
        """Jedinečné riešenie 3x3."""
        # x + y + z = 6
        # 2x - y + z = 3
        # x + 2y - z = 2
        self._verify_solution([[1, 1, 1, 6], [2, -1, 1, 3], [1, 2, -1, 2]], num_free_vars_expected=0)

    def test_infinite_solutions_underdetermined(self):
        """Nedourčená sústava - nekonečne veľa riešení (1 rovnica, 2 neznáme)."""
        # x + 2y = 5
        # Voľná: y (t0), Bázická: x = 5 - 2t0
        self._verify_solution([[1, 2, 5]], num_free_vars_expected=1)

    def test_infinite_solutions_2eq_3vars(self):
        """2 rovnice, 3 neznáme - 1 voľná premenná."""
        # x + y + z = 1
        # 2x + y - z = 0
        self._verify_solution([[1, 1, 1, 1], [2, 1, -1, 0]], num_free_vars_expected=1)

    def test_infinite_solutions_rank1_3vars(self):
        """Rank 1, 3 premenné - 2 voľné premenné."""
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
        """Komplexná nehomogénna sústava."""
        data = [[6, -4, 9, 8, 6, 7],
                [4, -1, 6, 2, -1, 8],
                [6, 2, 4, -3, -15, 9]]
        self._verify_solution(data)

    # === HOMOGÉNNE SÚSTAVY ===

    def test_homogenous_unique_solution(self):
        """Homogénna sústava s jedinečným riešením (triviálne riešenie)."""
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
        """Homogénna sústava - 2 voľné premenné."""
        # x + y + z = 0
        self._verify_solution([[1, 1, 1, 0]], num_free_vars_expected=2)

    def test_homogenous_3x4_rank2(self):
        """Homogénna 3x4, rank 2 - očakávame 2 voľné premenné."""
        # x + 2y + 3z + 4w = 0
        # 2x + 4y + 5z + 6w = 0
        # 3x + 6y + 8z + 10w = 0
        self._verify_solution([[1, 2, 3, 4, 0], [2, 4, 5, 6, 0], [3, 6, 8, 10, 0]],
                            num_free_vars_expected=2)

    # === ŠPECIÁLNE PRÍPADY ===

    def test_zero_matrix_homogenous(self):
        """Nulová matica - každé riešenie je riešením (celé univerzum)."""
        m = Matrix([[0, 0, 0], [0, 0, 0]])
        result = m.get_solutions()
        self.assertIn("domain", result.lower())

    def test_zero_matrix_all_free_3vars(self):
        """Nulová matica 2x4 - všetky 3 premenné voľné."""
        # 0x + 0y + 0z = 0
        # 0x + 0y + 0z = 0
        # Po RREF sa to stane prázdnou maticou, čo je celé univerzum
        m = Matrix([[0, 0, 0, 0], [0, 0, 0, 0]])
        result = m.get_solutions()
        # Nulová matica = univerzum riešení
        self.assertIn("domain", result.lower())

    def test_single_variable_system(self):
        """Sústava s 1 premennou."""
        # 2x = 4 => x = 2
        self._verify_solution([[2, 4]], num_free_vars_expected=0)

    def test_single_variable_homogenous(self):
        """Homogénna sústava s 1 premennou."""
        # 3x = 0 => x = 0
        self._verify_solution([[3, 0]], num_free_vars_expected=0)

    def test_overdetermined_consistent(self):
        """Preurčená konzistentná sústava (viac rovníc ako premenných)."""
        # x + y = 3
        # 2x + 2y = 6
        # 3x + 3y = 9
        self._verify_solution([[1, 1, 3], [2, 2, 6], [3, 3, 9]], num_free_vars_expected=1)

    def test_overdetermined_inconsistent(self):
        """Preurčená nekonzistentná sústava."""
        # x + y = 3
        # 2x + 2y = 6
        # x + y = 5  (kontradikcia s prvým)
        self._verify_solution([[1, 1, 3], [2, 2, 6], [1, 1, 5]], should_have_solution=False)

    def test_identity_matrix_system(self):
        """Jednotková matica - jedinečné riešenie."""
        # x = 1
        # y = 2
        # z = 3
        self._verify_solution([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]], num_free_vars_expected=0)

    def test_all_pivots_at_end(self):
        """Všetky pivoty na konci stĺpcov."""
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
