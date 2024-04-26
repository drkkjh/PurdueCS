from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.figure import Figure


class PyTorchANN(nn.Module):
    """
    A PyTorch implementation of a three-layer neural network with ReLU
    activation.

    Attributes:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units.
        output_size (int): The number of output units.
        model (nn.Sequential): The neural network model.
    """

    def __init__(self,
                    input_size: int,
                    hidden_size: int,
                    output_size: int,
                    random_seed: int = 42) -> None:
        """
        Initialize a PyTorchANN object.
        
        Parameters:
            input_size (int): The number of input features.
            hidden_size (int): The number of hidden units.
            output_size (int): The number of output units.
            random_seed (int): The random seed to use for PyTorch.

        Returns:
            None

        Examples:
            >>> pytorch_ann = PyTorchANN(input_size=20, output_size=10, hidden_size=10)
            >>> pytorch_ann.input_size
            20
            >>> pytorch_ann.hidden_size
            10
            >>> pytorch_ann.output_size
            10
            >>> pytorch_ann.seed
            42
            >>> pytorch_ann.model
            Sequential(
              (0): Linear(in_features=20, out_features=10, bias=True)
              (1): ReLU()
              (2): Linear(in_features=10, out_features=10, bias=True)
            )
        """
        # Call the parent class init, which makes PyTorch happy
        super(PyTorchANN, self).__init__()

        # Set the attributes
        # >>> YOUR CODE HERE >>>
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # <<< END OF YOUR CODE <<<
        
        # Set the random seed
        # >>> YOUR CODE HERE >>>
        self.seed = random_seed
        # <<< END OF YOUR CODE <<<
        torch.manual_seed(random_seed)

        # Create the model
        # >>> YOUR CODE HERE >>>
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
        # <<< END OF YOUR CODE <<<

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network. This method is called when you call
        the neural network as a function, e.g. pytorch_ann(X). i.e.
        pytorch_ann.forward(X) is called. It should return the output of the
        neural network.

        Parameters:
            X (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        Examples:
            >>> pytorch_ann = PyTorchANN(input_size=5, output_size=3, hidden_size=10)
            >>> pytorch_ann(torch.rand(5))
            tensor([-0.0675,  0.0238, -0.0057]...)
        """
        # >>> YOUR CODE HERE >>>
        output = self.model(X)
        # <<< END OF YOUR CODE <<<
        return output

    def train(self,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                X_val: torch.Tensor,
                y_val: torch.Tensor,
                epochs: int,
                batch_size: int,
                learning_rate: float) -> None:
        
        # Define the loss function and optimizer
        # >>> YOUR CODE HERE >>>
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        # <<< END OF YOUR CODE <<<

        train_losses = []
        train_accs = []

        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            # Prepare the batches for training
            for X_batch, y_batch in zip(X_train.split(batch_size),
                                        y_train.split(batch_size)):
                # Set the model to train mode
                self.model.train()

                # Zero the gradients for this iteration
                optimizer.zero_grad()

                # Forward pass through the network
                # >>> YOUR CODE HERE >>>
                output = self.forward(X_batch)
                # <<< END OF YOUR CODE <<<
                
                # Calculate the loss
                # >>> YOUR CODE HERE >>>
                loss = loss_function(output, y_batch.argmax(dim=1))
                # <<< END OF YOUR CODE <<<
                
                # Backpropagation
                # >>> YOUR CODE HERE >>>
                loss.backward()
                # <<< END OF YOUR CODE <<<

                # Update the parameters
                # >>> YOUR CODE HERE >>>
                optimizer.step()
                # <<< END OF YOUR CODE <<<

            # Set the model to eval mode
            self.model.eval()
            # Use torch.no_grad() to prevent tracking history (and using memory)
            # since backpropagation is not needed for validation
            with torch.no_grad():
                train_losses.append(loss_function(self.forward(X_train), y_train).item())
                train_accs.append(self.accuracy(X_train, y_train))

                val_losses.append(loss_function(self.forward(X_val), y_val).item())
                val_accs.append(self.accuracy(X_val, y_val))

            # Print the loss and accuracy for every 50th epoch
            if (epoch+1) % 5  == 0:
                print(f"[{epoch+1:} / {epochs}] | Train Loss: {train_losses[-1]:.5f} | Train Accuracy: {train_accs[-1]:.5f} | Val Loss: {val_losses[-1]:.5f} | Val Accuracy: {val_accs[-1]:.5f}")
            
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
            >>> pytorch_ann = PyTorchANN(input_size=5, output_size=3, hidden_size=10)
            >>> X = torch.rand((5, 5))
            >>> pytorch_ann.predict(X)
            tensor([1, 2, 2, 2, 2])
        """

        # Set the model to eval mode and use torch.no_grad()
        self.model.eval()
        with torch.no_grad():
            # predict the class labels
            # >>> YOUR CODE HERE >>>
            labels = self.forward(X).argmax(dim=1)
            # <<< END OF YOUR CODE <<<
            return labels

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
        # >>> YOUR CODE HERE >>>
        ax[0].plot(self.train_val_metrics["train_losses"], label="Train Loss")
        ax[0].plot(self.train_val_metrics["val_losses"], label="Val Loss")
        # <<< END OF YOUR CODE <<<

        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # Plot the training and validation accuracies
        # >>> YOUR CODE HERE >>>
        ax[1].plot(self.train_val_metrics["train_accs"], label="Train Accuracy")
        ax[1].plot(self.train_val_metrics["val_accs"], label="Val Accuracy")
        # <<< END OF YOUR CODE <<<

        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        fig.suptitle("Train/Val Metrics")
        fig.tight_layout()

        plt.savefig("pytorch_ann_train_val_metrics.png")

        return fig, ax
    
    def report_my_judgement(self) -> str:
        """
        Return your judgement on whether the neural network is overfitting or not.
        **You should not write any other code in this method.**

        Returns:
            str: Your judgement ("overfitting", "underfitting", or "generalizing")
        
        """

        # >>> YOUR CODE HERE >>>
        my_judgement = "overfitting"
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

        print_blue(f"Using device: {device}\n")

        X_train, y_train, X_test, y_test = load_mnist_f(return_tensor=True)

        # Sample 2000 points for training and 200 points for testing
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=2000, test_size=200, random_state=42)
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=200, test_size=1, random_state=42)

        # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], 784).float().to(device)
        X_val = X_val.reshape(X_val.shape[0], 784).float().to(device)
        X_test = X_test.reshape(X_test.shape[0], 784).float().to(device)

        # Normalize the images
        X_train = X_train / 255
        X_val = X_val / 255
        X_test = X_test / 255

        # One-hot encode the labels
        y_train = F.one_hot(y_train).float().to(device)
        y_val = F.one_hot(y_val).float().to(device)
        y_test = F.one_hot(y_test).float().to(device)

        # Train the neural network
        pytorch_ann = PyTorchANN(input_size=784, output_size=10, hidden_size=32).to(device)
        pytorch_ann.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=100,
            batch_size=64,
            learning_rate=0.2
        )
            

        test_accuracy = pytorch_ann.accuracy(X_test, y_test)

        pytorch_ann.plot_train_val_metrics()
        
        assert_greater_equal(test_accuracy, 0.8, f"\nAccuracy on test set: {pytorch_ann.accuracy(X_test, y_test):.5f}\n")

        print_blue(f"My judgement: {pytorch_ann.report_my_judgement()}\n")
    else:
        print_red("\nDoctests failed!\n")