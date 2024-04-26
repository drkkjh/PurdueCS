import numpy as np
from sklearn.model_selection import train_test_split

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    print("tqdm not found. Install it to see the progress bar if you want.")
    tqdm = lambda x: x

class LogisticRegression:
    """
    This class implements the Logistic Regression classifier.

    Attributes:
        w (np.ndarray): The weights of the model. The dimension of the weight
            vector is (n + 1, 1), where n is the number of features (columns)
            in the training data (X matrix). The bias term is included in the
            weight vector as the first element. Initialized to `None`.
        X (np.ndarray): The training data with bias term included. The dimension
            of the matrix is (m, n + 1), where m is the number of training
            examples and n is the number of features (columns) in the training
            data. Plus one is for the bias term (first column of ones). Initialized
            to `None`.
        y (np.ndarray): The labels for the training data. The dimension of the
            matrix is (m, 1), where m is the number of training examples.
            Initialized to `None`.
    """
    
    def __init__(self) -> None:
        """
        The constructor for LogisticRegression class. Initializes the weights,
        training data and labels to `None`.

        Parameters:
            None

        Returns:
            None

        Examples:
            >>> lr = LogisticRegression()
            >>> lr.w
            >>> lr.X
            >>> lr.y
        """

        self.w = None
        self.X = None
        self.y = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        This function applies the sigmoid function to `z`.
        
        Parameters:
            z (np.ndarray): The input to the sigmoid function.
        
        Returns:
            np.ndarray: The output of the sigmoid function.
        
        Examples:
            >>> lr = LogisticRegression()
            >>> lr.sigmoid(np.array([0, 1, 2]))
            array([0.5       , 0.73105858, 0.88079708])
        """

        # >>> YOUR CODE HERE >>>
        h = 1 / (1 + np.exp(-z))
        # <<< END OF YOUR CODE <<<
        
        return h
    
    def initialize_weights(self, cols: int) -> None:
        """
        This function initializes the weights to one.

        The dimension of the weight vector is (n + 1, 1), where n is the number
        of features in the training data (X matrix). The bias term is included
        in the weight vector as the first element.
        
        Parameters:
            cols (int): The number of columns in the training data (X matrix).
        
        Returns:
            None
        
        Examples:
            >>> lr = LogisticRegression()
            >>> lr.initialize_weights(3)
            >>> lr.w
            array([[1.],
                   [1.],
                   [1.]])
        """
        
        # >>> YOUR CODE HERE >>>
        self.w = np.ones(shape=(cols, 1))
        # <<< END OF YOUR CODE <<<
        
    
    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function computes the gradient of the loss function with respect
        to the weights.
        
        Parameters:
            X (np.ndarray): The training data of dimension (m, n + 1), where m
                is the number of training examples and n is the number of
                features (columns) in the training data. Plus one is for the
                bias term (first column of ones).
            y (np.ndarray): The labels for the training data of dimension (m, 1),
                where m is the number of training examples.
        
        Returns:
            np.ndarray: The gradient of the loss function with respect to the
                weights of dimension (n + 1, 1), where n is the number of
                features (columns) in the training data.
        
        Examples:
            >>> lr = LogisticRegression()
            >>> lr.initialize_weights(3)
            >>> X = np.array([[1, 1, 2], [1, 3, 4]])
            >>> y = np.array([[0, 1]]).T
            >>> lr.compute_gradient(X, y)
            array([[0.49083922],
                   [0.49050387],
                   [0.98134309]])
        """

        assert self.w is not None, "Initialize the weights first!"

        # Compute the gradient
        # >>> YOUR CODE HERE >>>
        m = X.shape[0]
        grad = (1 / m) * (X.T @ (self.sigmoid(X @ self.w) - y))
        # <<< END OF YOUR CODE <<<
        return grad

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            lr: float = 0.001,
            epochs: int = 500,
            verbose: bool = True) -> None:

        """
        This function trains the logistic regression model using gradient
        descent.

        Parameters:
            X (np.ndarray): The training data.
            y (np.ndarray): The labels for the training data.
            lr (float): The learning rate for gradient descent.
            epochs (int): The number of iterations to run gradient descent.

        Returns:
            None

        Examples:
            >>> lr = LogisticRegression()
            >>> X = np.array([[1, 1, 2], [1, 3, 4]])
            >>> y = np.array([[0, 1]]).T
            >>> lr.fit(X, y, lr=0.001, epochs=500, verbose=False)
            >>> lr.w
            array([[0.76029983],
                   [0.76128059],
                   [0.52158043]])
        """

        # Initialize the weights
        # >>> YOUR CODE HERE >>>
        self.initialize_weights(X.shape[1])
        # <<< END OF YOUR CODE <<<

        # Save X and y to the object
        self.X = X
        self.y = y

        assert self.w is not None, "Initialize the weights before fitting."

        # Run gradient descent
        for i in tqdm(range(epochs)):
            # Compute the gradient
            # >>> YOUR CODE HERE >>>
            grad = self.compute_gradient(X, y)
            # <<< END OF YOUR CODE <<<

            # Update the weights
            # >>> YOUR CODE HERE >>>
            self.w = self.w - lr * grad
            # <<< END OF YOUR CODE <<<

            if verbose and (i + 1) % 100 == 0:
                train_accuracy = self.accuracy(self.predict(X), y)
                tqdm.write(f"Epoch {i + 1:3d}/{epochs} | Train Accuracy: {train_accuracy:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This function predicts the labels for the data `X`.

        Parameters:
            X (np.ndarray): The data to predict the labels for.

        Returns:
            np.ndarray: The predicted labels for the data `X`.

        Examples:
            >>> lr = LogisticRegression()
            >>> X = np.array([[1, 2], [3, 4]])
            >>> lr.w = np.array([-1.99999999,  2.99999998])
            >>> prediction = lr.predict(X) # array([1, 1])
            >>> np.array_equal(prediction, np.array([1, 1]))
            True
        """

        assert self.w is not None, "Fit the model before predicting."

        # >>> YOUR CODE HERE >>>
        y_pred = (self.sigmoid(X @ self.w) >= 0.5).astype(int)
        # <<< END OF YOUR CODE <<<
        return y_pred
    
    def accuracy(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        This function computes the accuracy of the model.

        Parameters:
            y (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            float: The accuracy of the model.

        Examples:
            >>> lr = LogisticRegression()
            >>> y = np.array([0, 1])
            >>> y_pred = np.array([0, 1])
            >>> lr.accuracy(y, y_pred)
            1.0
        """

        # >>> YOUR CODE HERE >>>
        acc = (y == y_pred).sum() / y.shape[0]
        # <<< END OF YOUR CODE <<<
        
        return acc


if __name__ == "__main__":
    import doctest
    import os
    import warnings

    from utils import load_hw2_pickle, print_green, print_red

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # Suppress warnings
    warnings.filterwarnings('ignore')

    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")

        X, y = load_hw2_pickle(os.path.join(os.path.dirname(__file__), "train.pkl"))
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)

        lr = LogisticRegression()
        lr.fit(X_train, y_train, lr=0.001, epochs=20000, verbose=False)
        
        train_acc = lr.accuracy(lr.predict(X_train), y_train)
        valid_acc = lr.accuracy(lr.predict(X_valid), y_valid)
        
        if train_acc >= 0.70 and valid_acc >= 0.65:
            print_green(f"\nTraining accuracy: {lr.accuracy(lr.predict(X), y):.4f}, validation accuracy: {valid_acc:.4f}\noverall test passed!\n")
        else:
            print_red(f"\nTraining accuracy: {lr.accuracy(lr.predict(X), y):.4f}, validation accuracy: {valid_acc:.4f}\noverall test failed!\n")
    else:
        print_red("\nDoctests failed!\n")
