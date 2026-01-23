import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier
    )


def train_logistic_regression(X_train, y_train, preprocessor_pipeline):
    """
    Configure and train a Logistic Regression model with Lasso (L1)
    using GridSearchCV to optimize hyperparameters.
    """

    # 1. Define the base Logistic Regression model
    logit_base = LogisticRegression(
        solver='saga',
        l1_ratio=1.0,
        max_iter=5000,
        random_state=9999
    )

    # 2. Create the final Pipeline: Preprocessing + Model
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", logit_base)
    ])

    # 3. Define Hyperparameters
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    # 4. Optimization with GridSearchCV
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # 5. Model fitting
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_logistic_regression_poly(X_train, y_train, preprocessor_pipeline):
    """
    Train a Logistic Regression with Polynomial Features.
    """
    # 1. Define the base Logistic Regression model
    logit_base = LogisticRegression(
        solver='saga',
        l1_ratio=1.0,
        max_iter=15000,
        random_state=9999
    )

    # 2. Create the final Pipeline: Preprocessing + Model
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("poly", PolynomialFeatures(include_bias=False)),
        ("classifier", logit_base)
    ])

    # 3. Define Hyperparameters
    param_grid = {
        # Polynomial feature degrees
        'poly__degree': [1, 2],

        # Lasso regularization strength
        'classifier__C': [0.001, 0.1, 1, 10, 100],

        # Feature creation options
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def save_model(model, model_name, path="models"):
    """Save the best model using joblib (.pkl)"""
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, f"{model_name}.pkl")
    joblib.dump(model, file_path)
    return file_path


def update_experiment_log(model_name, best_params, metrics, folder="outputs"):
    """
    Record the results of the experiment in a CSV file.
    """
    # 1. Define the folder where the log will be saved
    file_path = os.path.join(folder, "experiment_log.csv")

    # 2. Meke sure that the destination folder exists
    os.makedirs(folder, exist_ok=True)

    # 3. Set the new row
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "best_params": str(best_params),
    }

    # 4. Combine with the received metrics
    log_entry.update(metrics)

    df_new = pd.DataFrame([log_entry])

    # 5. If the file does not exist, create it with header. If it exists, append without header.
    if not os.path.exists(file_path):
        df_new.to_csv(file_path, index=False)
    else:
        df_new.to_csv(file_path, mode='a', header=False, index=False)

    print(f"Experiment saved in {file_path}")


def train_sgd_logit(X_train, y_train, preprocessor_pipeline):
    """
    Train an SGDClassifier configured with loss='log' to emulate
    a Logistic Regression, optimized with Lasso (L1).
    """
    # 1. Define the SGDClassifier model
    sgd_logit_base = SGDClassifier(
        loss='log_loss',
        penalty='l1',
        max_iter=1000,
        tol=1e-3,
        random_state=9999
    )

    # 2. Create the final Pipeline: Preprocessing + Model
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", sgd_logit_base)
    ])

    # 3. Define hyperparameters
    param_grid = {
        'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
        'classifier__learning_rate': ['optimal', 'adaptive'],
        'classifier__eta0': [0.01, 0.1],
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    # 4. GridSearchCV Optimization
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # 5. Model fitting
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_sgd_svm(X_train, y_train, preprocessor_pipeline):
    """
    Train an SGDClassifier configured with loss='hinge' to emulate a linear
    Support Vector Machine (SVM), optimized with Lasso (L1).
    """
    # 1. Define the SGDClassifier model
    sgd_svm_base = SGDClassifier(
        loss='hinge',
        penalty='l1',
        max_iter=2000,
        tol=1e-3,
        random_state=9999
    )

    # 2. Create the final Pipeline: Preprocessing + Model
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", sgd_svm_base)
    ])

    # 3. Define hyperparameters
    param_grid = {
        'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
        'classifier__learning_rate': ['optimal', 'adaptive'],
        'classifier__eta0': [0.01, 0.1],
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    # 4. GridSearchCV Optimization
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # 5. Model fitting
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_svm_kernel(X_train, y_train, preprocessor_pipeline):
    """
    Train a nonlinear SVM (RBF Kernel) by optimizing C and gamma.
    """
    # 1. Define the base SVM model with RBF Kernel
    svc_base = SVC(kernel='rbf', random_state=9999)

    # 2. Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", svc_base)
    ])

    # 3. Define hyperparameters
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 0.1, 0.01, 0.001],
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    # 4. GridSearchCV Optimization
    print("Optimizing SVM model with RBF Kernel...")
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def train_svm_poly(X_train, y_train, preprocessor_pipeline):
    """
    Train a polynomial kernel SVM using the 'Kernel Trick'.
    Optimize degree, coef0, and C.
    """
    # 1. Define the base SVM model with Polynomial Kernel
    svc_poly = SVC(kernel='poly', random_state=9999, max_iter=150000, tol=1e-3)

    # 2. Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", svc_poly)
    ])

    # 3. Define hyperparameters
    param_grid = {
        'classifier__degree': [1, 2, 3, 4],
        'classifier__coef0': [0, 1, 5],
        'classifier__C': [0.1, 1, 10, 100],
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    # 4. GridSearchCV Optimization
    print("Optimizing SVM model with Polynomial Kernel...")
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # 5. Training
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def train_decision_tree(X_train, y_train, preprocessor_pipeline):
    """
    Train an optimized DecisionTreeClassifier using GridSearchCV.
    Regularization hyperparameters are applied to prevent overfitting.
    """
    # 1. Define the base Decision Tree model
    dt_base = DecisionTreeClassifier(random_state=9999)

    # 2. Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", dt_base)
    ])

    # 3. Define hyperparameters
    param_grid = {
        # Tree structure control
        'classifier__max_depth': [None, 3, 5, 10],
        'classifier__min_samples_leaf': [1, 5, 10, 20],
        'classifier__min_samples_split': [2, 10, 25, 50],

        # Quality metric
        'classifier__criterion': ['gini', 'entropy'],

        # Balance managed
        'classifier__class_weight': [None, 'balanced'],

        # Feature Engineering
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    # 4. GridSearchCV Optimization
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # 5. Training
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_