import pandas as pd


def print_red(msg):
    print("\033[91m {}\033[00m" .format(msg))


def print_green(msg):
    print("\033[92m {}\033[00m" .format(msg))


def print_yellow(msg):
    print("\033[93m {}\033[00m" .format(msg))


def assert_less_equal(actual, expected, msg):
    if actual < expected:
        print_green(msg)
    else:
        print_red(msg)


def load_hw3_csv(filename):
    df = pd.read_csv(filename)

    # drop Loan_ID column
    df = df.drop(columns=["Loan_ID"])

    # convert numerical columns to categorical
    numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for column in numerical_columns:
        df[column] = pd.cut(df[column], bins=5, labels=[
                            "low", "medium-low", "medium", "medium-high", "high"])

    # drop rows with missing values
    df = df.dropna()

    # convert y to binary
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y
