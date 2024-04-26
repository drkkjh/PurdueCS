from typing import List, Tuple

import numpy as np
from bagging import Bagging
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.ensemble import BaseEnsemble

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(x): return x


class CrossValidation():
    """
    Cross validation.

    Attributes:
        k (int): Number of folds.
        seed (int): Random seed for reproducibility.    
    """

    def __init__(self, k: int = 5, random_seed: int = 42) -> None:
        """
        Constructor for CrossValidation. Sets the number of folds and the random seed.

        Parameters:
            k (int): Number of folds.
            random_seed (int): Random seed for reproducibility.

        Returns:
            None

        Example:
            >>> cv = CrossValidation(k=5)
            >>> cv
            CrossValidation(k=5, seed=42)
        """

        assert k > 1, "k must be greater than 1."

        # >>> YOUR CODE HERE >>>
        self.k = k
        self.random_seed = random_seed
        # <<< END OF YOUR CODE <<<

    def split(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Split the data into k folds.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: X_folds, y_folds

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> y = np.array([0, 1, 0, 1])
            >>> X_folds, y_folds = cv.split(X, y)
            >>> X_folds
            [array([[3, 4]]), array([[7, 8]]), array([[1, 2]]), array([[5, 6]])...]
            >>> y_folds
            [array([1]), array([1]), array([0]), array([0])...]
        """

        assert len(X) == len(y), "X and y must have the same length."
        assert self.k <= len(
            X), "k cannot be larger than the number of samples."

        # Use np.random.seed(self.random_seed) to set the random seed.
        # >>> YOUR CODE HERE >>>
        np.random.seed(self.random_seed)
        # <<< END OF YOUR CODE <<<

        # Shuffle the data using np.random.permutation.
        # >>> YOUR CODE HERE >>>
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        # <<< END OF YOUR CODE <<<

        # Split the data into k folds using np.array_split.
        # >>> YOUR CODE HERE >>>
        X_folds = np.array_split(X, self.k)
        y_folds = np.array_split(y, self.k)
        # <<< END OF YOUR CODE <<<


        return X_folds, y_folds

    def _combine_fold(self,
                      X_folds: List[np.ndarray],
                      y_folds: List[np.ndarray],
                      i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use the ith fold as the test (validation) set and the rest as the training set.

        Parameters:
            X_folds (List[np.ndarray]): Features split into k folds.
            y_folds (List[np.ndarray]): Labels split into k folds.
            i (int): Index of the test (validation) fold.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X_folds = [np.array([[1, 2]]), np.array([[3, 4]]), np.array([[5, 6]]), np.array([[7, 8]])]
            >>> y_folds = [np.array([0]), np.array([1]), np.array([0]), np.array([1])]
            >>> cv._combine_fold(X_folds, y_folds, 0)
            (array([[3, 4],
                   [5, 6],
                   [7, 8]]), array([1, 0, 1]), array([[1, 2]]), array([0]))
        """

        # Use np.concatenate to combine the folds into the training set and the test set.
        # >>> YOUR CODE HERE >>>
        X_train = np.concatenate([X_folds[j] for j in range(self.k) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(self.k) if j != i])
        # <<< END OF YOUR CODE <<<


        # Assign the ith fold to be the test set.
        # >>> YOUR CODE HERE >>>
        X_test = X_folds[i]
        y_test = y_folds[i]
        # <<< END OF YOUR CODE <<<

        return X_train, y_train, X_test, y_test

    def score(self, X: np.ndarray, y: np.ndarray, model: BaseEnsemble) -> float:
        """
        Get the accuracy score of the model on the given data.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            model (BaseEnsemble): Model to evaluate.

        Returns:
            float: Accuracy score.

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> y = np.array([0, 1, 0, 1])
            >>> model = Bagging(2)
            >>> model.fit(X, y)
            >>> cv.score(X, y, model)
            0.75
        """
        # >>> YOUR CODE HERE >>>
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y)
        # <<< END OF YOUR CODE <<<

        return accuracy

    def cross_validate(self, X: np.ndarray, y: np.ndarray, model: BaseEnsemble) -> Tuple[float, List[float]]:
        """
        Perform k-fold cross validation on the given model and data.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            model (BaseEnsemble): Model to evaluate.

        Returns:
            float: Average accuracy score across all folds.
            List[float]: Accuracy scores for each fold.

        Example:
            (Run this file or submit to Gradescope to run the test)
        """

        # Split the data into k folds.
        # >>> YOUR CODE HERE >>>
        X_folds, y_folds = self.split(X, y)
        # <<< END OF YOUR CODE <<<
        scores = []

        # For each fold, train the model on the training set and evaluate on the test set.
        for i in range(self.k):
            # Get the training set and the test set, using the ith fold as the test set.
            # >>> YOUR CODE HERE >>>
            X_train, y_train, X_test, y_test = self._combine_fold(X_folds, y_folds, i)
            # <<< END OF YOUR CODE <<<

            # Train the model on the training set.
            # >>> YOUR CODE HERE >>>
            model.fit(X_train, y_train)
            # <<< END OF YOUR CODE <<<

            # Evaluate the model on the test set, and append the score to the scores list.
            # >>> YOUR CODE HERE >>>
            scores.append(self.score(X_test, y_test, model))
            # <<< END OF YOUR CODE <<<

        # Return the average score across all folds.
        return np.mean(scores), scores

    def get_best_model(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       params: List[int]) -> Tuple[BaseEnsemble, int, List[float]]:
        """
        Perform k-fold cross validation on the given model and data, and return the best model.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            model (BaseEnsemble): Model to evaluate.
            params (List[int]): List of parameters to try.

        Returns:
            BaseEnsemble: Best model.
            int: Best parameter.
            scores (List[float]): Accuracy scores for each parameter.

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> y = np.array([0, 1, 1, 1])
            >>> params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> model, best_n, scores = cv.get_best_model(X, y, params)
            >>> best_n
            1
            >>> scores
            [0.75, 0.5, 0.5, 0.25, 0.75, 0.75, 0.75, 0.75, 0.5, 0.5]
        """

        assert len(params) > 0, "params must be a non-empty list"

        # Store the parameters.
        self.params = params

        self.scores = []
        self.models = []

        # For each number of learners, train the model on the training set
        # and evaluate on the test set.
        for n in tqdm(params):
            # Create a new model with the given number of learners.
            # >>> YOUR CODE HERE >>>
            model = Bagging(n_learners=n)
            # <<< END OF YOUR CODE <<<

            # Perform k-fold cross validation on the model.
            # >>> YOUR CODE HERE >>>
            score, scores = self.cross_validate(X, y, model)
            # <<< END OF YOUR CODE <<<

            # Append the score to the scores list.
            self.scores.append(score)

            # Append the model to the models list.
            self.models.append(model)

        # Get the best number of learners and the corresponding model.
        # >>> YOUR CODE HERE >>>
        best_n_learners = self.params[np.argmax(self.scores)]
        best_model = self.models[np.argmax(self.scores)]
        # <<< END OF YOUR CODE <<<

        return best_model, best_n_learners, self.scores

    def plot_learning_curve(self) -> Figure:
        """
        Plot the learning curve for the best model.

        Parameters:
            None

        Returns:
            Matplotlib figure: Learning curve.

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> y = np.array([0, 1, 1, 1])
            >>> params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> model, best_n, scores = cv.get_best_model(X, y, params)
            >>> cv.plot_learning_curve() # should also save the figure to learning_curve.png
            <Figure size ... with 1 Axes>
        """

        assert self.models is not None, "get_best_model must be called before plot_learning_curve"
        assert self.scores is not None, "get_best_model must be called before plot_learning_curve"
        assert self.params is not None, "get_best_model must be called before plot_learning_curve"

        plt.figure()

        # Plot the learning curve.
        # >>> YOUR CODE HERE >>>
        plt.plot(self.params, self.scores)
        plt.xlabel("Number of Learners")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve")
        plt.savefig("learning_curve.png")
        # <<< END OF YOUR CODE <<<

        # plt.plot(self.params, self.scores)
        # plt.xlabel("Number of Learners")
        # plt.ylabel("Accuracy")
        # plt.title("Learning Curve")
        # plt.savefig("learning_curve.png")

        return plt.gcf()

    def __repr__(self) -> str:
        return f"CrossValidation(k={self.k}, seed={self.random_seed})"


if __name__ == "__main__":
    import doctest
    import os

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

        cv = CrossValidation(k=5)

        # with bagging
        best_bagging, best_n, scores = cv.get_best_model(
            X_train, y_train, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        print(f"Bagging | Best number of learners: {best_n}")
        print(f"Bagging | Average accuracy score: {np.mean(scores):.4f}\n")

        y_pred = best_bagging.predict(X_train)
        print(
            f"Bagging | Train accuracy: {accuracy_score(y_train, y_pred):.4f}")

        y_pred = best_bagging.predict(X_test)
        print(
            f"Bagging |  Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

        cv.plot_learning_curve()

    else:
        print_red("\nDoctests failed!\n")
