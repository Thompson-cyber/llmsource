# Description: This file contains the test cases generate by copilot for the merge sort algorithm with fault injection
# create four test cases for the merge sort algorithm
import unittest
import fault_injection_merge_sort as main


class TestMergeSort(unittest.TestCase):

    # Add four testCase methods for the merge sort algorithm to test the code with different input values
    # check if input is a list of integers
    # check if input is  mixed list of negative and positive integers
    # check if input is a float
    # check if input is a string

    def test_merge_sort(self):
        self.assertEqual(main.merge_sort_func([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])

    def test_merge_sort_mixed(self):
        self.assertEqual(main.merge_sort_func([-1, 2, -3, 4, -5]), [-5, -3, -1, 2, 4])

    def test_merge_sort_float(self):
        self.assertEqual(main.merge_sort_func([1.5, 2.5, 3.5, 4.5, 5.5]), [1.5, 2.5, 3.5, 4.5, 5.5])

    def test_merge_sort_string(self):
        self.assertEqual(main.merge_sort_func(["a", "b", "c", "d", "e"]), ["a", "b", "c", "d", "e"])


if __name__ == '__main__':
    unittest.main()
