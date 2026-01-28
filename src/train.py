import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
    )
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

        # Handling Imbalance
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

    # 5. Model fitting
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_random_forest(X_train, y_train, preprocessor_pipeline):
    """
    Configure and train a Random Forest Classifier model using GridSearchCV to optimize key hyperparameters.
    """

    # 1. Define the base Random Forest model
    rf_base = RandomForestClassifier(random_state=9999, n_jobs=-1)

    # 2. Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", rf_base)
    ])

    # 3. Define hyperparameters
    param_grid = {
        # Forest structure control
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_features': ['sqrt'],

        # Tree structure control
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_leaf': [1, 3, 5],

        # Handling Imbalance
        'classifier__class_weight': [None, 'balanced', 'balanced_subsample'],

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
        n_jobs=-1,
        verbose=1
    )

    # 5. Model fitting
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_extra_trees(X_train, y_train, preprocessor_pipeline):
    """
    Train an Extra-Trees (Extremely Randomized Trees) model.
    Unlike Random Forest, this model selects split thresholds randomly,
    reducing variance at the cost of a slight increase in bias.
    """

    # 1. Define the base Extra-Trees model
    et_base = ExtraTreesClassifier(random_state=9999, n_jobs=-1)

    # 2. Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", et_base)
    ])

    # 3. Define hyperparameters
    param_grid = {
        # Forest structure control
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_features': ['sqrt'],
        'classifier__bootstrap': [False, True],

        # Tree structure control
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_leaf': [1, 3, 5],

        # Handling Imbalance
        'classifier__class_weight': [None, 'balanced', 'balanced_subsample'],

        # Feature Engineering Flags
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
        n_jobs=-1,
        verbose=1
    )

    # 5. Model fitting
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_adaboost(X_train, y_train, preprocessor_pipeline):
    """
    Train an AdaBoost model using Decision Trees (Stumps).
    """

    # 1. Define base estimator: Decision Tree Stump
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=9999)

    # 2. Define base AdaBoost model
    ada = AdaBoostClassifier(
        estimator=base_estimator,
        random_state=9999
    )

    # 3. Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", ada)
    ])

    # 4. Define hyperparameters
    param_grid = {
        # Number of weak learners (trees)
        'classifier__n_estimators': [100, 300, 500],

        # Learning rate
        'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0],

        # Tree depth for base estimators
        'classifier__estimator__max_depth': [1, 2],

        # Feature Engineering
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    # 5. GridSearchCV Optimization
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    # 6. Model fitting
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_gradient_boosting(X_train, y_train, preprocessor_pipeline):
    """
    Train a Gradient Boosting model with built-in Early Stopping.
    """

    # 1. Define the base Gradient Boosting model with Early Stopping
    gb_base = GradientBoostingClassifier(
        n_estimators=1000,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=0.0001,
        random_state=9999
    )

    # 2. Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", gb_base)
    ])

    # 3. Define hyperparameters
    param_grid = {
        # Learning rate control
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],

        # Trees structure control
        'classifier__max_depth': [1, 2, 3, 4, 5],

        # Use a subsample of the data for each tree to reduce variance
        'classifier__subsample': [0.6, 0.8, 1.0],

        # Feature Engineering Flags
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
        n_jobs=-1,
        verbose=1
    )

    # 5. Model fitting
    grid_search.fit(X_train, y_train)

    # 6. Additional information about Early Stopping
    best_n_estimators = grid_search.best_estimator_.named_steps['classifier'].n_estimators_
    print(f"The best model stopped (Early Stopping) at {best_n_estimators} trees.")

    return grid_search.best_estimator_, grid_search.best_params_


def train_lightgbm(X_train, y_train, preprocessor_pipeline):
    """
    Train a LightGBM model using the Scikit-Learn API.
    """

    # 1. Define the base LightGBM model
    lgbm_base = LGBMClassifier(
        random_state=9999,
        n_jobs=-1,
        importance_type='gain',
        verbose=-1,
        bagging_freq=1
    )

    # 2. Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", lgbm_base)
    ])

    # 3. Define hyperparameters
    param_dist = {
        # Trees structure control
        'classifier__n_estimators': randint(100, 500),
        'classifier__num_leaves': randint(3, 31),
        'classifier__max_depth': [3, 5, -1],

        # Learning rate control
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],

        # Regularization L1 (Lasso) and L2 (Ridge)
        'classifier__reg_alpha': [0, 0.1, 0.5, 1.0],  # L1
        'classifier__reg_lambda': [0, 0.1, 0.5, 1.0],  # L2

        # Use a subsample of the data for each tree to reduce variance
        'classifier__bagging_fraction': [0.6, 0.8, 1.0],

        # Handling Imbalance
        'classifier__is_unbalance': [True, False],

        # Feature Engineering
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    # 4. RandomizedSearchCV Optimization
    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_dist,
        n_iter=250,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=9999
    )

    # 5. Model fitting
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_


def train_xgboost(X_train, y_train, preprocessor_pipeline):
    """
    Train an optimized XGBoost model with RandomizedSearchCV.
    Includes imbalance handling and strong regularization.
    """

    # 1. Calculate imbalance ratio for scale_pos_weight
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    epsilon = 1e-6
    scale_pos_weight = n_neg / (n_pos + epsilon)

    print(f"Imbalance ratio calculated for XGBoost: {scale_pos_weight:.2f}")

    # 2. Define the XGBoost base model
    xgb_base = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        random_state=9999,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )

    # 3. Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", xgb_base)
    ])

    # 4. Define hyperparameters
    param_dist = {
        # Number of trees
        'classifier__n_estimators': randint(50, 500),

        # Learning rate
        'classifier__learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1],

        # Trees structure control
        'classifier__max_depth': randint(3, 7),
        'classifier__min_child_weight': randint(1, 6),
        'classifier__gamma': [0, 0.1, 0.2, 0.5],

        # Use a subsample of the data for each tree to reduce variance (Bagging and Subsampling)
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],

        # Regularization L1 (Lasso) and L2 (Ridge)
        'classifier__reg_alpha': [0, 0.01, 0.1, 1, 10],
        'classifier__reg_lambda': [1, 1.5, 2, 5, 10],

        # Feature Engineering
        'preprocessor__feature_creator__Total_Income': [True, False],
        'preprocessor__feature_creator__Income_Loan_Ratio': [True, False],
        'preprocessor__feature_creator__High_Income_Flag': [True, False]
    }

    # 5. RandomizedSearchCV Optimization
    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_dist,
        n_iter=10000,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=9999
    )

    # 6. Model fitting
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_


def train_stacking_ensemble(X_train, y_train, preprocessor_pipeline, best_params_dict):
    """
    Train a StackingClassifier using the best models.
    """

    # 1. Define the best models
    estimators = [
        ('et', ExtraTreesClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=1,
            bootstrap=False,
            max_features='sqrt',
            random_state=9999
        )),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=9999
        )),
        ('xgb', XGBClassifier(
            n_estimators=353,
            learning_rate=0.001,
            max_depth=3,
            min_child_weight=2,
            gamma=0.5,
            colsample_bytree=1.0,
            subsample=1.0,
            reg_alpha=10,
            reg_lambda=1.5,
            random_state=9999,
            n_jobs=-1,
            tree_method='hist'
        )),
        ('logit', LogisticRegression(
            solver='saga',
            l1_ratio=1.0,
            C=10,
            max_iter=10000,
            random_state=9999
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=119,
            learning_rate=0.01,
            num_leaves=19,
            max_depth=3,
            reg_alpha=0.5,
            reg_lambda=0.5,
            bagging_fraction=1.0,
            is_unbalance=True,
            random_state=9999,
            n_jobs=-1,
            verbose=-1
        ))
    ]

    # 2. Define the Meta-Model
    final_estimator = LogisticRegression(
        l1_ratio=0,
        C=1.0,
        max_iter=10000,
        random_state=9999
    )

    # 3. Build the Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    # 4. Pipeline Completo
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline),
        ("classifier", stacking_clf)
    ])

    # 5. Model fitting
    full_pipeline.fit(X_train, y_train)

    return full_pipeline, stacking_clf.get_params()