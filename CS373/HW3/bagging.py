import numpy as np
from encoder import OneHotEncoder
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier


class Bagging(BaseEnsemble):
    """
    Bagging classifier with DecisionTreeClassifier as base learner.

    Attributes:
        learners: List of DecisionTreeClassifier objects.
        random_seed: Random seed for reproducibility.
        encoder: Encoder used to encode the data.
    """

    def __init__(self, n_learners: int, encoder: str = "onehot", random_seed: int = 42) -> None:
        """
        Constructor for Bagging classifier. Creates a list of DecisionTreeClassifier
        objects, sets the random seed, and initializes the encoder.

        Parameters:
            n_learners (int): Number of learners to train.

        Returns:
            None

        Example: 
            >>> bagging = Bagging(n_learners=10)
            >>> bagging
            Bagging(n_learners=10)
            >>> bagging.learners[0]
            DecisionTreeClassifier(random_state=42)
            >>> bagging.random_seed
            42
            >>> bagging.encoder
            OneHotEncoder()
        """

        # >>> YOUR CODE HERE >>>
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.learners = [DecisionTreeClassifier(random_state=random_seed) for _ in range(n_learners)]
        # np.random.seed(...)
        # <<< END OF YOUR CODE <<<

        if encoder == "onehot":
            # >>> YOUR CODE HERE >>>
            self.encoder = OneHotEncoder()
            # <<< END OF YOUR CODE <<<
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

    def _prepare_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare a sample of data for a learner. This method is used to create a
        bootstrap sample with replacement.

        Parameters:
            X (np.ndarray): Features.

        Returns:
            np.ndarray: Indices of the sample.

        Example:
            >>> bagging = Bagging(n_learners=10)
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> bagging._prepare_sample(X)
            array([2, 0, 2])
        """

        # >>> YOUR CODE HERE >>>
        indices = np.random.choice(len(X), len(X), replace=True)
        # <<< END OF YOUR CODE <<<
        return indices

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Bagging classifier.

        Parameters:
            X (np.ndarray): Training examples.
            y (np.ndarray): Training labels.

        Returns:
            None

        Example:
            >>> bagging = Bagging(n_learners=10)
            >>> X = np.array([ \
                    ["A", "1", "XX"], \
                    ["B", "2", "YY"], \
                    ["B", "2", "NA"], \
                    ["A", "1", "XX"], \
                    ["B", "2", "XX"], \
                ])
            >>> y = np.array([1, 1, 1, 0, 0])
            >>> bagging.fit(X, y)
            >>> X_encoded = bagging.encoder.encode(X)
            >>> bagging.learners[0].predict(X_encoded)
            array([0, 0, 1, 0, 0])
        """

        # encode the samples using the self.encoder
        # >>> YOUR CODE HERE >>>
        X_encoded = self.encoder.fit(X).encode(X)
        # <<< END OF YOUR CODE <<<

        # Train each learner on a bootstrap sample using the encoded data
        for learner in self.learners:
            # >>> YOUR CODE HERE >>>
            sample = self._prepare_sample(X_encoded)
            learner.fit(X_encoded[sample], y[sample])
            # learner.fit(...)
            # <<< END OF YOUR CODE <<<

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given examples.

        Parameters:
            X (np.ndarray): Examples to predict.

        Returns:
            np.ndarray: Predicted labels.

        Example:
            >>> bagging = Bagging(n_learners=10)
            >>> X = np.array([ \
                    ["A", "1", "XX"], \
                    ["B", "2", "YY"], \
                    ["B", "2", "NA"], \
                    ["A", "1", "XX"], \
                    ["B", "2", "XX"], \
                ])
            >>> y = np.array([1, 1, 1, 0, 0])
            >>> bagging.fit(X, y)
            >>> bagging.predict(X)
            array([0, 1, 1, 0, 0]...)
        """

        assert self.encoder.categories is not None, "Bagging classifier has not been fit yet."

        # encode the samples using the self.encoder
        # >>> YOUR CODE HERE >>>
        X_encoded = self.encoder.encode(X)
        # <<< END OF YOUR CODE <<<

        # Predict the labels for each learner
        # >>> YOUR CODE HERE >>>
        predictions = np.array([learner.predict(X_encoded) for learner in self.learners])
        # <<< END OF YOUR CODE <<<

        # Count the number of votes for each label
        # >>> YOUR CODE HERE >>>
        counts = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        # <<< END OF YOUR CODE <<<

        # Return the label with the most votes
        # >>> YOUR CODE HERE >>>
        y_pred = counts
        # <<< END OF YOUR CODE <<<
        return y_pred

    def __repr__(self) -> str:
        """
        String representation of Bagging classifier.

        Parameters:
            None

        Returns:
            str: String representation of Bagging classifier.
        """
        return f"Bagging(n_learners={len(self.learners)})"


if __name__ == "__main__":
    import doctest
    import os

    from encoder import OneHotEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from utils import load_hw3_csv, print_green, print_red

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")

        X, y = load_hw3_csv(os.path.join(
            os.path.dirname(__file__), "loan_train.csv"))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        bagging = Bagging(n_learners=10)
        bagging.fit(X_train, y_train)
        y_pred = bagging.predict(X_train)
        print(
            f"Bagging | Train accuracy: {accuracy_score(y_train, y_pred):.4f}")

        y_pred = bagging.predict(X_test)
        print(
            f"Bagging |  Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

        # without bagging
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit(X_train).encode(X_train)
        X_test_encoded = encoder.encode(X_test)
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_encoded, y_train)
        y_pred = dt.predict(X_train_encoded)
        print(
            f"Decision Tree | Train accuracy: {accuracy_score(y_train, y_pred):.4f}")
        y_pred = dt.predict(X_test_encoded)
        print(
            f"Decision Tree |  Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    else:
        print_red("\nDoctests failed!\n")
