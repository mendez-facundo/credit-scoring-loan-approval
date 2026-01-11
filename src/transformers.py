import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        # Check if it is a DataFrame
        if not hasattr(X, "columns"):
            raise TypeError("The input must be a Pandas' DataFrame.")

        # Save the original feature names
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        # Check if it is a DataFrame again in transform
        if not hasattr(X, "columns"):
            raise TypeError("The input must be a Pandas' DataFrame in order to drop columns.")

        return X.drop(self.columns_to_drop, axis=1, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        return np.array([f for f in input_features if f not in self.columns_to_drop])


class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, Total_Income=True, Income_Loan_Ratio=True, High_Income_Flag=True):
        self.Total_Income = Total_Income
        self.Income_Loan_Ratio = Income_Loan_Ratio
        self.High_Income_Flag = High_Income_Flag
        self.high_income_threshold = 0

    def fit(self, X, y=None):
        # 1. Check if X is a DataFrame
        if not hasattr(X, "columns"):
            raise TypeError("The input must be a Pandas' DataFrame.")

        # 2. Check necessary columns
        required_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in the DataFrame: {missing_cols}")

        # 3. Learning parameters and inspection
        temp_total_inc = X['ApplicantIncome'] + X['CoapplicantIncome']
        self.high_income_threshold_ = temp_total_inc.quantile(0.75)
        self.feature_names_in_ = np.array(X.columns)  # Names backup

        return self

    def transform(self, X):
        # Check if it is a DataFrame again in transform
        if not hasattr(X, "columns"):
            raise TypeError("The input must be a Pandas' DataFrame to transform.")

        X_copy = X.copy()
        total_inc = X_copy['ApplicantIncome'] + X_copy['CoapplicantIncome']

        # Dynamic feature creation
        if self.Total_Income:
            X_copy['Total_Income'] = total_inc
        if self.Income_Loan_Ratio:
            X_copy['Income_Loan_Ratio'] = np.where(
                (X_copy['LoanAmount'] > 0),
                total_inc / X_copy['LoanAmount'],
                0.0
            )
        if self.High_Income_Flag:
            X_copy['High_Income_Flag'] = (total_inc > self.high_income_threshold).astype(int)

        # 2. Dynamic column dropper
        # Only drop the original columns if at least one new feature was created
        if self.Total_Income or self.Income_Loan_Ratio or self.High_Income_Flag:
            cols_to_drop = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
            X_copy = X_copy.drop(columns=cols_to_drop, errors='ignore')

        return X_copy

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features = self.feature_names_in_
            else:
                raise RuntimeError("The transformer must be fitted before calling get_feature_names_out().")

        # Start with the list of input features
        feature_names = list(input_features)

        # Add the names of the new columns if they were created]
        if self.Total_Income:
            feature_names.append('Total_Income')
        if self.Income_Loan_Ratio:
            feature_names.append('Income_Loan_Ratio')
        if self.High_Income_Flag:
            feature_names.append('High_Income_Flag')

        # Delete the original columns if new features were created
        if self.Total_Income or self.Income_Loan_Ratio or self.High_Income_Flag:
            cols_to_drop = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
            feature_names = [f for f in feature_names if f not in cols_to_drop]

        return np.array(feature_names)