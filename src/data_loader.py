from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import os

def load_stratified_data(filepath=None, test_size=0.2, random_state=9999):
    # If no filepath is provided, use the default absolute path to the file
    if filepath is None:
        # Get the absolute path of the project directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        filepath = os.path.join(project_root, 'data', 'raw', 'loan_status.csv')

    data = pd.read_csv(filepath)

    # Map the target variable to numeric values
    data['Loan_Status'] = data['Loan_Status'].map({'N': 0, 'Y': 1})

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_index, test_index in split.split(data, data["Loan_Status"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    X_train = strat_train_set.drop("Loan_Status", axis=1)
    y_train = strat_train_set["Loan_Status"]
    X_test = strat_test_set.drop("Loan_Status", axis=1)
    y_test = strat_test_set["Loan_Status"]

    return X_train, X_test, y_train, y_test