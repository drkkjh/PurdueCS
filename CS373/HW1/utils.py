import numpy as np
import pandas as pd
from dt import DecisionTree


def print_red(msg):
    print("\033[91m {}\033[00m" .format(msg))

def print_green(msg):
    print("\033[92m {}\033[00m" .format(msg))

def print_yellow(msg):
    print("\033[93m {}\033[00m" .format(msg))

def read_hw1_data(filename):
	data = pd.read_csv(filename).fillna("NA").astype(str)
	y = data.iloc[:, 0].values
	X = data.iloc[:, 1:].values

	return X, y

def decision_tree_zero_one_loss(X_train, y_train, X_test, y_test, scorer, max_depth):
    tree = DecisionTree(scorer, max_depth=max_depth)
    tree.fit(X_train, y_train)

    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    train_loss = np.mean(y_train_pred != y_train)
    test_loss = np.mean(y_test_pred != y_test)

    return tree, train_loss, test_loss
