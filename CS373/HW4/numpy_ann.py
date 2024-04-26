import numpy as np


class NumpyANN():
    """
    A simple neural network with one hidden layer implemented using Numpy.

    Attributes:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units.
        output_size (int): The number of output units.
        lr (float): The learning rate.
        W1 (np.ndarray): The weights of the first layer.
        b1 (np.ndarray): The biases of the first layer.
        W2 (np.ndarray): The weights of the second layer.
        b2 (np.ndarray): The biases of the second layer.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 lr: float,
                 random_seed: int = 42) -> None:
        """
        Constructor for the NumpyANN class.

        Parameters:
            input_size (int): The number of input features.
            hidden_size (int): The number of hidden units.
            output_size (int): The number of output units.
            lr (float): The learning rate.
            random_seed (int): The random seed to use. Defaults to 42.

        Returns:
            None

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size=3, output_size=2, lr=0.01)
            >>> ann.input_size
            5
            >>> ann.hidden_size
            3
            >>> ann.output_size
            2
            >>> ann.lr
            0.01
            >>> ann.W2.shape
            (3, 2)
            >>> ann.b2.shape
            (1, 2)
        """
        
        # Set the random seed
        np.random.seed(random_seed)

        # Initialize attributes
        # >>> YOUR CODE HERE >>>
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        # <<< END OF YOUR CODE <<<
        
        # Initialize weight for the first layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))

        # Initialize weight for the second layer like the first layer
        # >>> YOUR CODE HERE >>>
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        # <<< END OF YOUR CODE <<<

    def _relu(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the ReLU activation function.

        Parameters:
            z (np.ndarray): The input to the ReLU function.

        Returns:
            np.ndarray: The output of the ReLU function.
            
        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size=3, output_size=2, lr=0.01)
            >>> ann._relu(np.array([-1, 0, 1]))
            array([0, 0, 1])
        """
        # >>> YOUR CODE HERE >>>
        relu = np.maximum(0, z)
        # <<< END OF YOUR CODE <<<
        return relu
    
    def _relu_grad(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the ReLU activation function.

        Parameters:
            z (np.ndarray): The input to the ReLU function.

        Returns:
            np.ndarray: The gradient of the ReLU function.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size=3, output_size=2, lr=0.01)
            >>> ann._relu_grad(np.array([-1, 0, 1]))
            array([0, 0, 1])
        """
        # >>> YOUR CODE HERE >>>
        relu_grad = np.where(z > 0, 1, 0)
        # <<< END OF YOUR CODE <<<
        return relu_grad
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the softmax activation function.

        Parameters:
            z (np.ndarray): The input to the softmax function.

        Returns:
            np.ndarray: The output of the softmax function.
        """

        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def _cross_entropy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the cross entropy loss.

        Parameters:
            y_pred (np.ndarray): The predicted probabilities for each class.
            y_true (np.ndarray): The ground truth labels (one-hot encoded)

        Returns:
            float: The cross entropy loss.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size=3, output_size=2, lr=0.01)
            >>> ann._cross_entropy(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[1, 0], [0, 1]]))
            0.69314...
        """
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Performs a forward pass through the network.

        Parameters:
            X (np.ndarray): The input to the network.
            training (bool): Whether the network is in training mode or not. Defaults to True.

        Returns:
            np.ndarray: The output of the network.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size=3, output_size=2, lr=0.01)
            >>> ann._forward(np.array([[1, 2, 3, 4, 5]]))
            array([[0.50013..., 0.49986...]])
        """
        
        # Forward pass, a = activation(z), z = Wx + b
        # Layer 1, use relu activation
        # >>> YOUR CODE HERE >>>
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        # <<< END OF YOUR CODE <<<

        # Layer 2, use softmax activation
        # >>> YOUR CODE HERE >>>
        z2 = a1 @ self.W2 + self.b2
        a2 = self._softmax(z2)
        # <<< END OF YOUR CODE <<<

        # Save the activations for backpropagation
        if training:
            self.z1 = z1
            self.a1 = a1
            self.z2 = z2
            self.a2 = a2

        return a2
    
    def _backward(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Performs a backward propagation through the network.

        Parameters:
            X (np.ndarray): The input to the network.
            y_true (np.ndarray): The ground truth labels (one-hot encoded)

        Returns:
            None

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size=3, output_size=2, lr=0.01)
            >>> ann._forward(np.array([[1, 2, 3, 4, 5]]))
            array([[0.50013134, 0.49986866]])
            >>> ann._backward(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 0]]))
            >>> ann.W2
            array([[-0.00504001, -0.01071118],
                   [ 0.00314247, -0.00908024],
                   [-0.01412304,  0.01465649]])
        """

        # Get the number of samples
        m = y_true.shape[0]

        # Backward pass
        # dL/dz2 = a2 - y_true
        dz2 = self.a2 - y_true

        # dL/dW2 = dL/dz2 * dz2/dW2, where dz2/dW2 = a1
        # Divide by m to take the average over all samples
        dW2 = np.dot(self.a1.T, dz2) / m

        # dL/db2 = dL/dz2 * dz2/db2, where dz2/db2 = 1, so dz2/db2 is just dz2
        # Summing up the gradients for each neuron in the layer
        # Divide by m to take the average over all samples
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # dL/dz1 = dL/dz2 * dz2/da1 * da1/dz1,
        # where dz2/da1 = W2, da1/dz1 = relu_grad(z1)
        # >>> YOUR CODE HERE >>>
        dz1 = dz2 @ self.W2.T * self._relu_grad(self.z1)
        # <<< END OF YOUR CODE <<<

        # dL/dW1 = dL/dz1 * dz1/dW1, where dz1/dW1 = X
        # dL/db1 = dL/dz1 * dz1/db1, where dz1/db1 = 1, so dz1/db1 is just dz1
        # Divide by m to take the average over all samples
        # >>> YOUR CODE HERE >>>
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        # <<< END OF YOUR CODE <<<

        # Perform gradient descent for layer 2
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Perform gradient descent for layer 1
        # >>> YOUR CODE HERE >>>
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        # <<< END OF YOUR CODE <<<

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int,
              batch_size: int) -> None:
        """
        Trains the neural network.

        Parameters:
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The training labels.
            X_val (np.ndarray): The validation data.
            y_val (np.ndarray): The validation labels.
            epochs (int): The number of epochs to train for.
            batch_size (int): The batch size to use for mini-batch gradient descent.

        Returns:
            None

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size=3, output_size=2, lr=0.01)
            >>> ann.train(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 0]]), np.array([[1, 2, 3, 4, 5]]), np.array([[1, 0]]), epochs=1, batch_size=1)
        """
        for epoch in range(epochs):
            # Prepare the batches for mini-batch gradient descent
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # Perform a forward pass
                # >>> YOUR CODE HERE >>>
                self._forward(X_batch, training=True)
                # <<< END OF YOUR CODE <<<

                # Perform a backward pass
                # >>> YOUR CODE HERE >>>
                self._backward(X_batch, y_batch)
                # <<< END OF YOUR CODE <<<

            y_pred_train = self._forward(X_train, training=False)
            y_pred_val = self._forward(X_val, training=False)

            # Compute the loss
            # >>> YOUR CODE HERE >>>
            train_loss = self._cross_entropy(y_pred_train, y_train)
            val_loss = self._cross_entropy(y_pred_val, y_val)
            # <<< END OF YOUR CODE <<<

            train_acc = self.accuracy(X_train, y_train)
            val_acc = self.accuracy(X_val, y_val)

            if (epoch+1) % 10 == 0:
                print(f"[{epoch+1:} / {epochs}] | Train Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.5f} | Val Loss: {val_loss:.5f} | Val Accuracy: {val_acc:.5f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the neural network.

        Parameters:
            X (np.ndarray): The input to the network.

        Returns:
            np.ndarray: The predicted probabilities for each class.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size=3, output_size=2, lr=0.01)
            >>> ann.predict(np.array([[1, 2, 3, 4, 5]]))
            array([[0.50013134, 0.49986866]])
        """

        # >>> YOUR CODE HERE >>>
        pred = self._forward(X, training=False)
        # <<< END OF YOUR CODE <<<
        return pred
    
    def accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the accuracy of the network.

        Parameters:
            X (np.ndarray): The input to the network.
            y_true (np.ndarray): The ground truth labels (one-hot encoded)

        Returns:
            float: The accuracy of the network.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size=3, output_size=2, lr=0.01)
            >>> ann.accuracy(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 0]]))
            1.0
        """

        # >>> YOUR CODE HERE >>>
        y_pred = self.predict(X)
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
        # <<< END OF YOUR CODE <<<
        return acc
    
if __name__ == "__main__":
    import doctest
    import os

    from sklearn.model_selection import train_test_split
    from utils import (assert_greater_equal, load_mnist_f, print_green,
                       print_red)

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")

        X_train, y_train, X_test, y_test = load_mnist_f(return_tensor=False)

        # Sample 2000 points for training and 200 points for testing
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=2000, test_size=200, random_state=42)
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=200, test_size=1, random_state=42)
        # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], 784)
        X_val = X_val.reshape(X_val.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)

        # Normalize the images
        X_train = X_train / 255
        X_val = X_val / 255
        X_test = X_test / 255

        # One-hot encode the labels
        y_train = np.eye(10)[y_train]
        y_val = np.eye(10)[y_val]
        y_test = np.eye(10)[y_test]

        # Train the neural network
        ann = NumpyANN(input_size=784, hidden_size=64, output_size=10, lr=0.01)
        ann.train(X_train, y_train, X_val, y_val, epochs=250, batch_size=64)

        test_accuracy = ann.accuracy(X_test, y_test)
        
        assert_greater_equal(test_accuracy, 0.77, f"\nAccuracy on test set: {ann.accuracy(X_test, y_test):.5f}\n")
    else:
        print_red("\nDoctests failed!\n")