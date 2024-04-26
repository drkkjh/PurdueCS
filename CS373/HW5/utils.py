import os

import matplotlib.pyplot as plt
import torchvision


def print_red(msg):
    print("\033[91m {}\033[00m" .format(msg))

def print_green(msg):
    print("\033[92m {}\033[00m" .format(msg))

def print_yellow(msg):
    print("\033[93m {}\033[00m" .format(msg))

def print_blue(msg):
    print("\033[94m {}\033[00m" .format(msg))
    
def assert_less_equal(actual, expected, msg):
    if actual < expected:
        print_green(msg)
    else:
        print_red(msg)

def assert_greater_equal(actual, expected, msg):
    if actual > expected:
        print_green(msg)
    else:
        print_red(msg)

def plot_explained_variance_ratio(explained_variance_ratio, filename: str = 'explained_variance_ratio.png') -> None:
    """
    Plot explained variance ratio of the data. The x-axis should be the number
    of components and the y-axis should be the cumulative explained variance 
    ratio. Save the plot to a file given by filename.

    Args:
        filename: The filename to save the plot to.

    Returns:
        None.
    """

    plt.plot(explained_variance_ratio,
             label='Explained Variance Ratio')

    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()

    plt.savefig(os.path.join(os.path.dirname(__file__), filename))


def compare_digit_images(filename, X, X_pca):
    fig, axs = plt.subplots(2, 5, figsize=(1.5*5, 2*2))

    for i in range(5):
        img = X[i].reshape((28, 28))
        axs[0, i].imshow(img, cmap='gray')

    for i in range(5):
        img = X_pca[i].reshape((28, 28))
        axs[1, i].imshow(img, cmap='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), filename))

def load_mnist_f(return_tensor=False):
    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)

    if return_tensor:
        X_train = mnist_train.data
        y_train = mnist_train.targets
        X_test = mnist_test.data
        y_test = mnist_test.targets
    else:
        X_train = mnist_train.data.numpy()
        y_train = mnist_train.targets.numpy()
        X_test = mnist_test.data.numpy()
        y_test = mnist_test.targets.numpy()

    return X_train, y_train, X_test, y_test
