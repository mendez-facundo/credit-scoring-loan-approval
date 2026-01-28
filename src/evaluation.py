import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import graphviz
from sklearn.tree import export_graphviz
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
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
    Extract and display the significance of the variables.
    Compatible with:
    - Linear models (Logistic Regression, SVM linear) -> uses .coef_
    - Tree-based models (Decision Tree, Random Forest, Gradient Boosting) -> uses .feature_importances_
    """
    # Extract feature names from the preprocessor step
    try:
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    except AttributeError:
        # Fallback for older scikit-learn versions or simple pipelines
        feature_names = [f"Feature_{i}" for i in range(model.named_steps['classifier'].n_features_in_)]

    classifier = model.named_steps['classifier']

    # Automatic detection of model type
    if hasattr(classifier, 'coef_'):
        # For linear models (Lasso, Ridge, Linear SVM)
        # Flatten is used because coef_ is shape (1, n_features) for binary classification
        importances = classifier.coef_.flatten()
        metric_name = 'Coefficient'

    elif hasattr(classifier, 'feature_importances_'):
        # For Trees and Ensembles (Random Forest, XGBoost, etc.)
        importances = classifier.feature_importances_
        metric_name = 'Importance'

    else:
        print("The model has no known attributes of importance (coef_ or feature_importances_)")
        return None

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        metric_name: importances
    })

    # Sorting logic:
    # - For coefficients: we care about magnitude (absolute value), but keep the sign.
    # - For importances: they are always positive, so abs() doesn't change anything but allows reusing logic.
    importance_df['Abs_Value'] = importance_df[metric_name].abs()
    importance_df = importance_df.sort_values(by='Abs_Value', ascending=False).drop(columns=['Abs_Value'])

    print(f"\n--- Feature {metric_name} ---")
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


def analyze_support_vectors(model, X_train, y_train, model_name="svm_model"):
    """
    Analyze and visualize the support vectors of an SVM (SVC) model.
    Requires the model to be a pipeline with 'preprocessor' and 'classifier' steps.
    """

    # 1. Get classifier and preprocessor from the pipeline
    try:
        clf = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
    except:
        print("Error: The model should be a pipeline with 'preprocessor' and 'classifier' steps.")
        return

    # Check if the classifier is an SVM with support vectors
    if not hasattr(clf, "support_vectors_"):
        print(f"The classifier {type(clf).__name__} doesn't have support vectors.")
        return

    # 2. Support Vectors Analysis
    n_total = X_train.shape[0]
    n_sv = clf.support_vectors_.shape[0]
    ratio = n_sv / n_total

    print(f"\n--- Support Vector Analysis for {model_name} ---")
    print(f"Total number of training instances: {n_total}")
    print(f"Total Support Vectors: {n_sv}")
    print(f"Ratio (SVs / Total): {ratio:.2%}")

    # Get indexes of support vectors
    sv_indices = clf.support_

    # Check distribution of classes in support vectors
    if hasattr(y_train, "iloc"):
        sv_labels = y_train.iloc[sv_indices]
    else:
        sv_labels = y_train[sv_indices]

    print("\nClass distribution in the Support Vectors:")
    print(sv_labels.value_counts())

    # 3. Visualization of Support Vectors using PCA
    print("\nGenerating PCA visualization...")

    # a) Transform training data (scaling/encoding)
    X_processed = preprocessor.transform(X_train)

    # b) Reduce to 2 dimensions to plot using PCA
    pca = PCA(n_components=2, random_state=9999)
    X_pca = pca.fit_transform(X_processed)

    # c) Transform to numpy array if necessary
    if hasattr(X_pca, "values"):
        X_pca = X_pca.values
    elif not isinstance(X_pca, np.ndarray):
        X_pca = np.array(X_pca)

    # d) Indentify support vectors in the PCA space
    sv_pca = X_pca[sv_indices]

    # e) Plot
    plt.figure(figsize=(10, 6))

    # Regular data points in light gray
    # Transform y_train to array if it's a Series or DataFrame
    y_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_arr,
                          cmap='coolwarm', alpha=0.3, label='Data Points')

    # Support Vectors as black empty circles
    plt.scatter(sv_pca[:, 0], sv_pca[:, 1], s=100,
                facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')

    plt.title(f"SVM Support Vectors (PCA Projection) - {model_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'outputs/{model_name}_support_vectors.png')
    plt.show()

    return ratio


def visualize_tree(model, preprocessor, feature_names, class_names=['Rejected', 'Approved'], path="outputs"):
    """
    Export the decision tree visualization to a .dot and .png file.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Remove the classifier from the pipeline if necessary
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['classifier']
    else:
        classifier = model

    dot_path = os.path.join(path, "decision_tree.dot")
    png_path = os.path.join(path, "decision_tree")  # graphviz adds the extension

    # 1. Export to DOT format
    export_graphviz(
        classifier,
        out_file=dot_path,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
        filled=True
    )
    print(f"Tree exported to: {dot_path}")

    # 2. Try to convert the DOT file to PNG
    try:
        source = graphviz.Source.from_file(dot_path)
        source.format = "png"
        source.render(filename="decision_tree", directory=path, cleanup=True)
        print(f"Image saved in: {png_path}.png")
    except Exception as e:
        print(f"The PNG image could not be generated (Check Graphviz binaries installation): {e}")
        print(f"You can view the .dot file online at http://viz-js.com/")