from typing import List, Tuple

import numpy as np
# Name: Derrick Khoo

class NumpyBasics:
    def split_list(self, x: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function takes a list of numbers and returns a tuple of two numpy
        arrays. The first array contains the first half of the list and the
        second array contains the second half. If the length of the list is odd,
        the first array should contain the extra item.

        Parameters:
            x (list): A list of numbers.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays.

        Examples:
            >>> nb = NumpyBasics()

            >>> x = [1, 2, 3, 4, 5, 6]
            >>> first_half, second_half = nb.split_list(x)
            >>> type(first_half)
            <class 'numpy.ndarray'>
            >>> type(second_half)
            <class 'numpy.ndarray'>
            >>> first_half
            array([1, 2, 3])
            >>> second_half
            array([4, 5, 6])

            >>> nb = NumpyBasics()
            >>> x = [1, 2, 3, 4, 5]
            >>> first_half, second_half = nb.split_list(x)
            >>> first_half
            array([1, 2, 3])
            >>> second_half
            array([4, 5])

        """

        # >>> YOUR CODE HERE >>>
        length = len(x)
        middle = length // 2
        firstLength = middle + (length % 2)
        first_half = np.array(x[:firstLength])
        second_half = np.array(x[firstLength:])
        # <<< END OF YOUR CODE <<<


        return first_half, second_half

    def multiply_inverse(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function takes two numpy arrays and returns a 2-D numpy array that
        contains the inverse of the dot product of the first array and the
        second array. Note that you will always be able to take the dot product
        of the x and y without reshaping. You may find the function
        `np.dot` and `np.linalg.inv` useful.

        Parameters:
            x (np.ndarray): A numpy array.
            y (np.ndarray): A numpy array.

        Returns:
            np.ndarray: A 2-D numpy array.

        Examples:
            >>> nb = NumpyBasics()

            >>> x = np.array([[1, 2, 3], [4, 5, 6]])
            >>> y = np.array([[1, 2], [3, 4], [5, 6]])
            >>> a = nb.multiply_inverse(x, y)
            >>> type(a)
            <class 'numpy.ndarray'>
            >>> a
            array([[ 1.77777778, -0.77777778],
                   [-1.36111111,  0.61111111]])

        """

        # >>> YOUR CODE HERE >>>
        dotProduct = np.dot(x, y)
        array = np.linalg.inv(dotProduct)
        # <<< END OF YOUR CODE <<<


        return array

    def custom_normalization(self, x: np.ndarray) -> np.ndarray:
        """
        This function takes a numpy array and returns a numpy array where all
        negative values are replaced with the minimum value in the array.
        After replacing the negative values, the new array is normalized by
        multiplying each value by the value at the same position in the
        original array. Return the square root of the normalized array.

        Parameters:
            x (np.ndarray): A numpy array.

        Returns:
            np.ndarray: A numpy array.

        Examples:
            >>> nb = NumpyBasics()

            >>> x = np.array([[-1, 2, 3], [4, -5, 6]])
            >>> a = nb.custom_normalization(x)
            >>> type(a)
            <class 'numpy.ndarray'>
            >>> a
            array([[2.23606798, 2.        , 3.        ],
                   [4.        , 5.        , 6.        ]])

        """

        # >>> YOUR CODE HERE >>>
        negIdx = np.where(x < 0)
        minimum = np.min(x)
        # print(minimum)
        original = x.copy()
        x[negIdx] = minimum
        normalizedX = x * original
        array = np.sqrt(normalizedX)
        # <<< END OF YOUR CODE <<<


        return array


if __name__ == "__main__":
    import doctest
    import os

    from utils import print_green, print_red

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # Run the doctests. If all tests pass, print "All tests pass!"
    # You may ignore PYDEV DEBUGGER WARNINGS that appear in the console.
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green("\nAll tests passed!\n")
    else:
        print_red("\nSome tests failed!\n")
