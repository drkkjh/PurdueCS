from abc import abstractmethod

import numpy as np


class Encoder:
    """
    Abstract class for encoding data.
    """
    def __init__(self, data: np.ndarray) -> None:
        """
        Constructor for Encoder class.

        Parameters:
            data (np.ndarray): Data to fit.

        Returns:
            None
        """
        self.data = data

    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Abstract method for encoding data.

        Parameters:
            data (np.ndarray): Data to encode.

        Returns:
            np.ndarray: Encoded data.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class OneHotEncoder(Encoder):
    def __init__(self):
        """
        Constructor for OneHotEncoder class.

        Parameters:
            None

        Returns:
            None

        Example:
            >>> encoder = OneHotEncoder()
            >>> encoder
            OneHotEncoder()
            >>> encoder.data
            >>> encoder.categories
            >>> encoder.encoded_data
        """

        # >>> YOUR CODE HERE >>>
        self.data = None
        self.categories = None
        self.encoded_data = None
        # <<< END OF YOUR CODE <<<
    
    def fit(self, data: np.ndarray) -> Encoder:
        """
        Fit the OneHotEncoder.

        Parameters:
            data (np.ndarray): Data to fit.

        Returns:
            Encoder: Fitted OneHotEncoder.

        Example:
            >>> encoder = OneHotEncoder()
            >>> X = np.array([["Male", 1], \
                              ["Female", 3], \
                              ["Female", 2]])
            >>> encoder.fit(X).categories[0]
            array(['Female', 'Male'], dtype=...)
            >>> encoder.categories[1]
            array(['1', '2', '3'], dtype=...)
        """

        # Save the data to self.data.
        # >>> YOUR CODE HERE >>>
        self.data = data
        # <<< END OF YOUR CODE <<<

        # Create an empty list to store the unique categories for each column.
        # >>> YOUR CODE HERE >>>
        self.categories = []
        # <<< END OF YOUR CODE <<<

        # Iterate over each column in the data and append the unique categories
        # to self.categories.
        for i in range(data.shape[1]):
            # >>> YOUR CODE HERE >>>
            self.categories.append(np.unique(data[:, i]))
            # <<< END OF YOUR CODE <<<

        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data using one-hot encoding.

        Parameters:
            data (np.ndarray): Data to encode.

        Returns:
            np.ndarray: Encoded data.

        Example:
            >>> encoder = OneHotEncoder()
            >>> X = np.array([["Male", 1], \
                              ["Female", 3], \
                              ["Female", 2]])
            >>> X_ = np.array([["Female", 1], \
                              ["Male", 4]])
            >>> encoder.fit(X).encode(X_)
            array([[1, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0]])
        """

        if self.categories is None:
            self.fit(data)

        encoded_data = []

        for i in range(data.shape[1]): # for each column in the input data

            # create a 2D array of zeros with the same number of rows as the
            # input data and a number of columns equal to the number of unique
            # categories in the current column.

            # use np.zeros and self.categories.
            # the unique categories in i-th column are stored in self.categories[i].

            # >>> YOUR CODE HERE >>>
            encoding_array = np.zeros((data.shape[0], len(self.categories[i])), dtype=int)
            # <<< END OF YOUR CODE <<<


            for j in range(data.shape[0]): # for each row in the current column

                # find the index of the category of the current cell in the
                # array of stored categories. np.where may be useful here.
                # >>> YOUR CODE HERE >>>
                category_index = np.where(self.categories[i] == data[j, i])
                # <<< END OF YOUR CODE <<<

                # set the cell in the one-hot encoded data at the corresponding row and category index to 1
                # >>> YOUR CODE HERE >>>
                encoding_array[j, category_index] = 1
                # <<< END OF YOUR CODE <<<

            # append the encoded column to the list of encoded data
            encoded_data.append(encoding_array)


        # concatenate the encoded data along the second axis
        encoded_data = np.concatenate(encoded_data, axis=1)

        # store the encoded data in the encoder object
        self.encoded_data = encoded_data

        return encoded_data
    
    def __repr__(self) -> str:
        return super().__repr__()
    


if __name__ == "__main__":
    import doctest
    import os

    from utils import print_green, print_red

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")
    else:
        print_red("\nDoctests failed!\n")