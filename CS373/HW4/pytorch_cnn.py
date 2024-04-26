from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.figure import Figure
import random

class PyTorchCNN(nn.Module):
    """
    A PyTorch convolutional neural network model.

    Attributes:
        num_classes (int): The number of classes.
        model (nn.Sequential): The model.
    """

    def __init__(self, num_classes: int = 10, random_seed: int = 42) -> None:
        """
        Initialize the PyTorchCNN model.

        Parameters:
            num_classes (int): The number of classes.
            random_seed (int): The random seed.

        Returns:
            None

        Examples:
            >>> pytorch_cnn = PyTorchCNN()
            >>> pytorch_cnn.num_classes
            10
            >>> pytorch_cnn.model
            Sequential...
        """
        # 
        super(PyTorchCNN, self).__init__()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.num_classes = num_classes

        # Define the model that runs on Fashion MNIST
        # >>> YOUR CODE HERE >>>
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features=self.num_classes)
        )
        # <<< END OF YOUR CODE <<<

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the neural network.
        """
        output = self.model(x)
        return output
    
    def train(self,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                X_val: torch.Tensor,
                y_val: torch.Tensor,
                epochs: int = 10,
                learning_rate: float = 0.001) -> None:
        """
        Train the neural network.

        Parameters:
            X_train (torch.Tensor): The training data.
            y_train (torch.Tensor): The training labels.
            X_val (torch.Tensor): The validation data.
            y_val (torch.Tensor): The validation labels.
            epochs (int): The number of epochs to train for.
            learning_rate (float): The learning rate.

        Returns:
            None
        """

        # Define the loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []


        for epoch in range(epochs):
            self.model.train()

            optimizer.zero_grad()

            output = self.forward(X_train)

            loss = loss_function(output, y_train)
            loss.backward()

            optimizer.step()

            train_loss = loss.item()
            train_losses.append(train_loss)
            train_accs.append(self.accuracy(X_train, y_train))

            self.model.eval()
            with torch.no_grad():
                val_losses.append(loss_function(self.forward(X_val), y_val).item())
                val_accs.append(self.accuracy(X_val, y_val))

            if (epoch+1) % 1 == 0:
                print(f"[{epoch+1:} / {epochs}] | Train Loss: {train_loss:.5f} | Train Accuracy: {train_accs[-1]:.5f} | Val Loss: {val_losses[-1]:.5f} | Val Accuracy: {val_accs[-1]:.5f}")
    
        self.train_val_metrics = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs
        }

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the class labels for the input data.
        
        Parameters:
            X (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted class labels.

        Examples:
            >>> pytorch_cnn = PyTorchCNN()
            >>> X = torch.randn(10, 1, 28, 28)
            >>> pytorch_cnn.predict(X)
            tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        """

        # Set the model to eval mode and use torch.no_grad()
        self.model.eval()
        with torch.no_grad():
            return self.forward(X).argmax(dim=1)

    def plot_train_val_metrics(self) -> Tuple[Figure, np.ndarray]:
        """
        Plot the training and validation metrics.

        Parameters:
            None

        Returns:
            Tuple[Figure, np.ndarray]: A tuple containing the matplotlib
                Figure and Axes objects.
        """

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the training and validation losses
        ax[0].plot(self.train_val_metrics["train_losses"], label="Train Loss")
        ax[0].plot(self.train_val_metrics["val_losses"], label="Val Loss")

        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # Plot the training and validation accuracies
        ax[1].plot(self.train_val_metrics["train_accs"], label="Train Accuracy")
        ax[1].plot(self.train_val_metrics["val_accs"], label="Val Accuracy")

        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        fig.suptitle("Train/Val Metrics")
        fig.tight_layout()

        plt.savefig("pytorch_cnn_train_val_metrics.png")

        return fig, ax
    
    def report_my_judgement(self) -> str:
        """
        Return your judgement on whether the neural network is overfitting or not.
        **You should not write any other code in this method.**
        
        Returns:
            str: Your judgement ("overfitting", "underfitting", or "generalizing")
        
        """

        # >>> YOUR CODE HERE >>>
        my_judgement = "generalizing"
        # <<< END OF YOUR CODE <<<

        assert my_judgement in ["overfitting", "underfitting", "generalizing"], \
            f"Your judgement must be one of 'overfitting', 'underfitting', or 'generalizing'. You entered: {my_judgement}\n"

        return my_judgement

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculate the accuracy of the neural network on the input data.

        Parameters:
            X (torch.Tensor): The input data.
            y (torch.Tensor): The true class labels.

        Returns:
            float: The accuracy of the neural network.
        """    
        assert X.shape[0] == y.shape[0], f"X.shape[0] != y.shape[0] ({X.shape[0]} != {y.shape[0]})"
        correct = torch.sum(self.predict(X) == y.argmax(dim=1))
        return (correct / X.shape[0]).item()


if __name__ == "__main__":
    import doctest
    import os

    from sklearn.model_selection import train_test_split
    from utils import (assert_greater_equal, load_mnist_f, print_blue,
                       print_green, print_red)

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            print_blue("GPU not available. Using CPU instead.")
        # device = torch.device("cpu")
        print_blue(f"Using device: {device}\n")

        X_train, y_train, X_test, y_test = load_mnist_f(return_tensor=True)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=2000, test_size=200, random_state=42)
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=200, test_size=1, random_state=42)

        # Flatten the images (weight of size [32, 1, 3, 3], expected input[1, 2000, 28, 28])
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).float().to(device)
        X_val = X_val.reshape(X_val.shape[0], 1, 28, 28).float().to(device)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).float().to(device)

        # Normalize the images
        X_train = X_train / 255
        X_val = X_val / 255
        X_test = X_test / 255

        # One-hot encode the labels
        y_train = F.one_hot(y_train).float().to(device)
        y_val = F.one_hot(y_val).float().to(device)
        y_test = F.one_hot(y_test).float().to(device)

        # Train the neural network
        pytorch_cnn = PyTorchCNN(num_classes=10).to(device)
        pytorch_cnn.train(X_train, y_train, X_val, y_val, epochs=60, learning_rate=0.002)
            

        test_accuracy = pytorch_cnn.accuracy(X_test, y_test)

        pytorch_cnn.plot_train_val_metrics()
        
        assert_greater_equal(test_accuracy, 0.85, f"\nAccuracy on test set: {pytorch_cnn.accuracy(X_test, y_test):.5f}\n")

        print_blue(f"My judgement: {pytorch_cnn.report_my_judgement()}\n")
    else:
        print_red("\nDoctests failed!\n")