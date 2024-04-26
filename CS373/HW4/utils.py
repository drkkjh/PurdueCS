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