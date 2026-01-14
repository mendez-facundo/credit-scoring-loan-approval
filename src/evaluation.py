import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)


def analyze_feature_importance(model):
    """
    Extract and display the significance of the variables (coefficients)
    of lineal model (such as Logistic Regression or SGD) within a pipeline.
    """
    # Best model created with a train.py function
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coefficients = model.named_steps['classifier'].coef_.flatten()

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)

    print("\n--- Feature Significance (Lasso) ---")
    print(importance_df.to_string(index=False))
    return importance_df


def get_optimal_threshold(model, X_test, y_test, target_precision=None):
    """
    Calculate the ideal threshold. If target_precision is defined, find the minimum
    threshold to achieve that precision. Otherwise, maximize the F1-Score. (For models
    that output probabilities, such as Logistic Regression or SGD with log loss)
    """
    # Get probabilities for the positive class
    y_probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

    # Calculate F1-Scores
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    if target_precision:
        # Look for the minimum threshold to achieve the target precision
        idx = np.argmax(precisions >= target_precision)
        best_threshold = thresholds[idx]
        best_f1 = f1_scores[idx]
        print(f"Criterion: Target precision of {target_precision * 100}%")
    else:
        # Maximize F1
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        print("Criterion: Maximizing F1-Score (Balance)")

    return best_threshold, best_f1, precisions, recalls, thresholds


def get_optimal_threshold_from_scores(model, X_test, y_test):
    """
    Calculate the ideal threshold for models that do NOT return probabilities,
    but decision scores (such as SVM with hinge loss), maximizing the F1-Score.
    """
    # 1. Get decision scores
    y_scores = model.decision_function(X_test)

    # 2. Calculate precision, recall and corresponding thresholds
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    # 3. Calculate F1-Scores
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    # 4. Find the best threshold maximizing F1
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Criterion: Maximizing F1-Score for Scores (SVM)")
    return best_threshold, best_f1, precisions, recalls, thresholds


def full_classification_report(model, X_test, y_test, threshold, model_name="model"):
    """Generate final metrics using a custom threshold."""
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_scores >= threshold).astype(int)

    # 1. Calculate metrics for logging
    metrics = {
        "threshold": round(threshold, 4),
        "roc_auc": round(roc_auc_score(y_test, y_scores), 4),
        "f1_score": round(f1_score(y_test, y_pred_custom), 4),
        "precision": round(precision_score(y_test, y_pred_custom), 4),
        "recall": round(recall_score(y_test, y_pred_custom), 4)
    }

    # 2. Show final metrics
    roc_auc = roc_auc_score(y_test, y_scores)
    print(f"\nFinal evaluation for {model_name}:")
    print(f" - Adjusted Threshold: {threshold:.4f}")
    print(f" - ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report (Adjusted):")
    print(classification_report(y_test, y_pred_custom))

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_custom)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title(f"Confusion Matrix: {model_name} (Th={threshold:.2f})")
    plt.xlabel("Prediction")
    plt.ylabel("Real Value")

    plt.savefig(f'outputs/{model_name}_cm_adjusted.png')
    plt.show()

    return metrics


def full_classification_report_from_scores(model, X_test, y_test, threshold, model_name="model"):
    """Generate final metrics using a threshold based on decision_function."""
    y_scores = model.decision_function(X_test)
    y_pred_custom = (y_scores >= threshold).astype(int)

    # 1. Metrics dictionary for logging
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    metrics = {
        "threshold": round(threshold, 4),
        "roc_auc": round(roc_auc_score(y_test, y_scores), 4),
        "f1_score": round(f1_score(y_test, y_pred_custom), 4),
        "precision": round(precision_score(y_test, y_pred_custom), 4),
        "recall": round(recall_score(y_test, y_pred_custom), 4)
    }

    # 2. Show final metrics
    roc_auc = roc_auc_score(y_test, y_scores)
    print(f"\nFinal evaluation for {model_name}:")
    print(f" - Adjusted Threshold: {threshold:.4f}")
    print(f" - ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report (Adjusted):")
    print(classification_report(y_test, y_pred_custom))

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_custom)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title(f"Confusion Matrix: {model_name} (Th={threshold:.2f})")
    plt.xlabel("Prediction")
    plt.ylabel("Real Value")

    plt.savefig(f'outputs/{model_name}_cm_adjusted.png')
    plt.show()

    return metrics


def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, scoring='f1'):
    """
    Generates and plots the learning curve to diagnose bias and variance.

    Args:
        estimator: The model or pipeline to be evaluated.
        X: Training features.
        y: Training target.
        title: Graph title.
        cv: Number of folds for cross-validation.
        scoring: Metric to be evaluated (default 'f1').
    """

    # Sample sizes for training: 10%, 32%, 55%, 78%, 100% of the dataset
    train_sizes = np.linspace(0.1, 1.0, 5)

    # Calculate learning curves using cross-validation
    train_sizes_abs, train_scores, validation_scores = learning_curve(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(validation_scores, axis=1)
    val_std = np.std(validation_scores, axis=1)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes_abs, val_mean, 'o-', color="g", label="Cross-Validation Score")

    # Draw error bands
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")

    plt.title(f"{title} - Metric: {scoring}")
    plt.xlabel("Training Set Size")
    plt.ylabel(f"{scoring} Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
