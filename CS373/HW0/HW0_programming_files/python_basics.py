import random
from typing import Dict, List, Union
# Name: Derrick Khoo


class Multiplication:
    """
    This class takes two numbers and multiplies them together.

    Attributes:
        first (Union[int, float]): The first number to multiply.
        second (Union[int, float]): The second number to multiply.
        answer (Union[int, float]): The product of the two numbers.
    """

    def __init__(self,
                 first: Union[int, float],
                 second: Union[int, float]) -> None:
        """
        The constructor for Multiplication class. Saves the first and second
        numbers to `self.first` and `self.second` respectively.

        Parameters:
            first (Union[int, float]): The first number to multiply.
            second (Union[int, float]): The second number to multiply.

        Returns:
            None

        Examples:
            >>> multiplication = Multiplication(2, 3)
            >>> multiplication.first
            2

            >>> multiplication.second
            3

            >>> multiplication.answer
            Traceback (most recent call last):
            ...
            AttributeError: 'Multiplication' object has no attribute 'answer'
        """

        # >>> YOUR CODE HERE >>>
        self.first = first
        self.second = second
        # <<< END OF YOUR CODE <<<


    def multiply(self) -> Union[int, float]:
        """
        This function multiplies the first and second numbers and returns the
        product.

        Parameters:
            None

        Returns:
            Union[int, float]: The product of the first and second numbers.

        Examples:
            >>> multiplication = Multiplication(1, 5)
            >>> multiplication.multiply()
            5
            >>> multiplication.answer
            5
        """

        # >>> YOUR CODE HERE >>>
        self.answer = self.first * self.second
        # <<< END OF YOUR CODE <<<


        return self.answer

    def display(self) -> None:
        """
        This function prints the first number and second number with 2 decimal 
        places and the product with 4 decimal places. You may find the `round`
        and `str` functions useful.

        print(f'str1 {var1} str2 {var2} ...') -> this is called f-strings and
        is a convenient way to print strings with variables.

        Parameters:
            None

        Returns:
            None

        Examples:
            >>> multiplication = Multiplication(1, 5)
            >>> multiplication.multiply()
            5
            >>> multiplication.display()
            First: 1.00
            Second: 5.00
            Product: 5.0000
        """

        # >>> YOUR CODE HERE >>>
        print(f'First: {self.first:.2f}')
        print(f'Second: {self.second:.2f}')
        print(f'Product: {self.answer:.4f}')
        # <<< END OF YOUR CODE <<<



class DataStructure:
    def generate_list(self) -> List[int]:
        """
        This function returns a list of 10 random numbers from 0 to 4 inclusive.

        Parameters:
            None

        Returns:
            List[int]: A list of any three different numbers from 0 to 4.

        Examples:
            >>> data_structure = DataStructure()
            >>> random.seed(0) # Do not put this line in your code!
            >>> data_structure.generate_list()
            [3, 3, 0, 2, 4, 3, 3, 2, 3, 2]
        """

        # >>> YOUR CODE HERE >>>
        # Do not include random.seed() in your code!
        my_list = [random.randint(0, 4) for _ in range(10)]
        # <<< END OF YOUR CODE <<<


        return my_list

    def find_unique(self, my_list: List[int]) -> set[int]:
        """
        This function takes a list of numbers and returns a set of unique
        numbers.

        Parameters:
            my_list (List[int]): A list of numbers.

        Returns:
            set[int]: A set of unique numbers.

        Examples:
            >>> data_structure = DataStructure()
            >>> random_list = [3, 3, 0, 2, 4, 3, 3, 2, 3, 2]
            >>> data_structure.find_unique(random_list)
            {0, 2, 3, 4}
        """

        # >>> YOUR CODE HERE >>>
        unique = set(my_list)
        # <<< END OF YOUR CODE <<<


        return unique

    def count_frequency(self, my_list: List[int]) -> Dict[int, int]:
        """
        This function takes a list of numbers and returns a dictionary of
        numbers and their frequencies.

        Parameters:
            my_list (List[int]): A list of numbers.

        Returns:
            Dict[int, int]: A dictionary of numbers and their frequencies.

        Examples:
            >>> data_structure = DataStructure()
            >>> random_list = [3, 3, 0, 2, 4, 3, 3, 2, 3, 2]
            >>> data_structure.count_frequency(random_list)
            {0: 1, 2: 3, 3: 5, 4: 1}
        """

        # >>> YOUR CODE HERE >>>
        frequency = {}
        for num in my_list:
            if num in frequency:
                frequency[num] += 1
            else:
                frequency[num] = 1
        # <<< END OF YOUR CODE <<<


        return dict(sorted(frequency.items()))


if __name__ == "__main__":
    import doctest
    import os
    import warnings
    from utils import print_green, print_red

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # Run the doctests. If all tests pass, print "All tests pass!"
    # You may ignore PYDEV DEBUGGER WARNINGS that appear in the console.
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green("\nAll tests passed!\n")
    else:
        print_red("\nSome tests failed!\n")
