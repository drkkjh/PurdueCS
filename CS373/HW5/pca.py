from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class PCA():
    """
    Principal Component Analysis (PCA).

    Attributes:
        n_components (int): The number of principal components to keep.
    """

    def __init__(self, n_components: int) -> None:
        """
        Constructor for PCA class.

        Parameters:
            n_components (int): The number of principal components to keep.

        Returns:
            None

        Examples:
            >>> pca = PCA(n_components=2)
            >>> pca.n_components
            2
        """
        
        # >>> YOUR CODE HERE >>>
        self.n_components = n_components
        # <<< END OF YOUR CODE <<<

    def cov(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the covariance matrix for the dataset X.

        Parameters:
            X (np.ndarray): The dataset of shape (N, D).

        Returns:
            cov (np.ndarray): The covariance matrix of shape (D, D).

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> pca = PCA(n_components=2)
            >>> pca.cov(X)
            array([[4., 4.],
                   [4., 4.]])

        """

        # Compute the covariance matrix
        # >>> YOUR CODE HERE >>>
        cov = np.cov(X, rowvar=False)
        # <<< END OF YOUR CODE <<<

        return cov
    
    def eig(self, cov_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the eigenvalues and corresponding eigenvectors
        for the covariance matrix cov_matrix.

        Parameters:
            cov_matrix (np.ndarray): The covariance matrix of shape (D, D).

        Returns:
            eigen_values (np.ndarray): The eigenvalues of shape (D,).
            eigen_vectors (np.ndarray): The eigenvectors of shape (D, D).

        Examples:
            >>> cov_matrix = np.array([[4, 4], [4, 4]])
            >>> pca = PCA(n_components=2)
            >>> eigen_values, eigen_vectors = pca.eig(cov_matrix)
            >>> eigen_values
            array([0., 8.])
            >>> eigen_vectors
            array([[-0.70710678,  0.70710678],
                   [ 0.70710678,  0.70710678]])
        """

        # Compute the eigenvalues and eigenvectors, use np.linalg.eigh
        # >>> YOUR CODE HERE >>>
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        # <<< END OF YOUR CODE <<<

        return eigen_values, eigen_vectors
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Trains the PCA model on the dataset X.

        Parameters:
            X (np.ndarray): The dataset of shape (N, D).

        Returns:
            self (PCA): The trained PCA model.

        Examples:
            >>> X = np.array([[1, 2], [3, 4]])
            >>> pca = PCA(n_components=1)
            >>> pca.fit(X)
            <__main__.PCA object at 0x...>
            >>> pca.components_
            array([[0.70710678],
                   [0.70710678]])
            >>> pca.explained_variance_ratio_
            array([1.])
            >>> pca.explained_variance_ratio_784
            array([1., 0.])
        """

        # Compute the covariance matrix
        # >>> YOUR CODE HERE >>>
        cov_matrix = self.cov(X)
        # <<< END OF YOUR CODE <<<

        # Compute the eigenvalues and eigenvectors
        # >>> YOUR CODE HERE >>>
        eigen_values, eigen_vectors = self.eig(cov_matrix)
        # <<< END OF YOUR CODE <<<

        # Sort the eigenvalues and eigenvectors, argsort may be useful
        # >>> YOUR CODE HERE >>>
        idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        # <<< END OF YOUR CODE <<<
        # Store the first n eigenvectors
        # >>> YOUR CODE HERE >>>
        self.components_ = eigen_vectors[:, :self.n_components]
        # <<< END OF YOUR CODE <<<

        # Store the explained variance ratio
        # >>> YOUR CODE HERE >>>
        self.explained_variance_ratio_ = eigen_values[:self.n_components] / np.sum(eigen_values)
        self.explained_variance_ratio_784 = eigen_values[:784] / np.sum(eigen_values)
        # <<< END OF YOUR CODE <<<

        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies dimensionality reduction to X using the trained model.

        Parameters:
            X (np.ndarray): The dataset of shape (N, D).

        Returns:
            X_transformed (np.ndarray): The transformed dataset of shape (N, n_components).

        Examples:
            >>> X = np.array([[1, 2], [3, 4]])
            >>> pca = PCA(n_components=1)
            >>> pca.fit(X)
            <__main__.PCA object at 0x...>
            >>> pca.transform(X)
            array([[-1.41421356],
                   [ 1.41421356]])
        """

        # Zero-center the data (subtract the mean)
        # >>> YOUR CODE HERE >>>
        X = X - np.mean(X, axis=0)
        # <<< END OF YOUR CODE <<<

        # Project the data onto the eigenvectors
        # >>> YOUR CODE HERE >>>
        X_transformed = X @ self.components_
        # <<< END OF YOUR CODE <<<

        return X_transformed
    
    def inverse_transform(self, X_transformed: np.ndarray, X_original: np.ndarray) -> np.ndarray:
        """
        Applies the inverse transformation to X.

        Parameters:
            X_transformed (np.ndarray): The transformed dataset of shape (N, n_components).

        Returns:
            X_original (np.ndarray): The reconstructed original dataset of shape (N, D).

        Examples:
            >>> X = np.array([[1, 2], [3, 4]])
            >>> pca = PCA(n_components=1)
            >>> pca.fit(X)
            <__main__.PCA object at 0x...>
            >>> X_transformed = pca.transform(X)
            >>> pca.inverse_transform(X_transformed, X)
            array([[1., 2.],
                   [3., 4.]])
        """

        # Invert the transformation, don't forget to add the mean back in
        # >>> YOUR CODE HERE >>>
        X_reconstructed = X_transformed @ self.components_.T + np.mean(X_original, axis=0)
        # <<< END OF YOUR CODE <<<


        return X_reconstructed

    def report_my_judgement(self) -> int:
        """
        Based ona the explained variance ratio, what might be a good number of
        components to keep?

        Parameters:
            None

        Returns:
            judgement (int): Your judgement as an integer.
        """

        # >>> YOUR CODE HERE >>>
        judgement = 10
        # <<< END OF YOUR CODE <<<
        assert isinstance(judgement, int)
        return judgement

if __name__ == "__main__":
    import doctest
    import os

    from sklearn.model_selection import train_test_split
    from utils import (assert_greater_equal, compare_digit_images,
                       load_mnist_f, plot_explained_variance_ratio,
                       print_green, print_red)

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")
        X_train, y_train, X_test, y_test = load_mnist_f(return_tensor=False)

        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=20000, random_state=42)
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=2000, random_state=42)

        # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)

        knn = KNeighborsClassifier()

        knn.fit(X_train, y_train)

        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)
        
        train_acc = sum(y_train == y_pred_train) / len(y_train)
        test_acc = sum(y_test == y_pred_test) / len(y_test)

        print(
            f"KNN's Performance before PCA: train_acc={train_acc}, test_acc={test_acc}")

        pca = PCA(n_components=50)
        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        knn.fit(X_train_pca, y_train)


        p_train_pca = knn.predict(X_train_pca)
        train_acc_pca = sum(y_train == p_train_pca) / len(y_train)

        p_test_pca = knn.predict(X_test_pca)
        test_acc_pca = sum(y_test == p_test_pca) / len(y_test)


        print(
            f"KNN's Performance after PCA: train_acc={train_acc_pca}, test_acc={test_acc_pca}")

        plot_explained_variance_ratio(pca.explained_variance_ratio_)

        X_train_inv = pca.inverse_transform(X_train_pca, X_train)
        compare_digit_images('compare.png', X_train, X_train_inv)
    else:
        print_red("\nDoctests failed!\n")