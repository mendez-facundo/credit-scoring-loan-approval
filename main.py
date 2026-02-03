import joblib
import os
from src.data_loader import load_stratified_data
from src.pipelines import get_preprocessing_pipeline, get_kpca_pipeline
from src.train import (
    save_model,
    update_experiment_log,
    train_logistic_regression,
    train_logistic_regression_poly,
    train_sgd_logit,
    train_sgd_svm,
    train_svm_kernel,
    train_svm_poly,
    train_decision_tree,
    train_random_forest,
    train_extra_trees,
    train_adaboost,
    train_gradient_boosting,
    train_lightgbm,
    train_xgboost,
    train_stacking_ensemble,
    train_kpca_logit
    )
from src.evaluation import (
    analyze_feature_importance,
    get_optimal_threshold,
    get_optimal_threshold_from_scores,
    full_classification_report,
    full_classification_report_from_scores,
    plot_learning_curve,
    analyze_support_vectors,
    visualize_tree
    )

def run_training_pipeline(model_name="logit_kpca_v1"):
    print(f"--- Initiating training for {model_name} ---")

    # 1. Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_stratified_data()

    # 2. Get preprocessing pipeline
    print("Setting up preprocessing pipeline...")
    preprocessor = get_kpca_pipeline()

    # 3. Model training and hyperparameter optimization
    print(f"Initiating training: {model_name}...")
    best_model, best_params = train_kpca_logit(X_train, y_train, preprocessor)
    print(f"Best hyperparameters: {best_params}")

    # 4. Significance analysis of variables
    # analyze_feature_importance(best_model)
    """
    try:
        # Access meta-model (Logistic Regression) inside the Stacking
        meta_model = best_model.named_steps['classifier'].final_estimator_
        base_model_names = best_model.named_steps['classifier'].named_estimators_.keys()

        # Extract the coefficients assigned by the meta-model to each base model
        if hasattr(meta_model, 'coef_'):
            print("\n--- Meta-Learner Weights (Base Model Importance) ---")
            meta_weights = meta_model.coef_.flatten()
            for name, weight in zip(base_model_names, meta_weights):
                print(f"Model {name}: {weight:.4f}")
        else:
            print("Meta-learner does not provide coefficients.")
    except Exception as e:
        print(f"Could not analyze meta-learner weights: {e}")
    """

    # 5. Threshold Tuning
    ideal_th, best_f1, _, _, _ = get_optimal_threshold(best_model, X_test, y_test)
    print(f"Ideal threshold: {ideal_th:.4f}")

    # 6. Model's final evaluation with the ideal threshold
    metrics_dict = full_classification_report(
        best_model,
        X_test,
        y_test,
        threshold=ideal_th,
        model_name=model_name
    )

    # 7. Save the model
    model_path = save_model(best_model, model_name)
    print(f"Model saved in: {model_path}")

    # 8. Register the experiment in CSV file
    update_experiment_log(model_name, best_params, metrics_dict)

    # 9. Learning curve analysis
    print("\nGenerating Learning Curves...")
    plot_learning_curve(
        best_model,
        X_train,
        y_train,
        title=f"Learning Curve - {model_name}",
        scoring='f1'
    )

def run_evaluation_pipeline(model_name="svm_rbf_v1"):
    print(f"--- Initiating evaluation for {model_name} ---")

    model_path = os.path.join("models", f"{model_name}.pkl")

    # 1. Confirm model existence
    if not os.path.exists(model_path):
        print(f"ERROR: the model wasn't find in {model_path}")
        print("First train the model using the training pipeline.")
        return

    # 2. Cargar datos y modelo
    print("Loading data and saved model..")
    X_train, X_test, y_train, y_test = load_stratified_data()
    best_model = joblib.load(model_path)

    # Precaution in case you saved the GridSearchCV object instead of the direct estimator
    if hasattr(best_model, 'best_estimator_'):
        best_model = best_model.best_estimator_
        print("-> Best estimator extracted from GridSearchCV object.")

    # 3. Analyze support vectors
    print(f"\nAttempting Support Vector Analysis for {model_name}...")
    try:
        analyze_support_vectors(best_model, X_train, y_train, model_name=model_name)
    except Exception as e:
        print(f"Skipping Support Vector Analysis: {e}")

    # 4. Learning curves
    print("Generating Learning Curves...")
    plot_learning_curve(
        best_model,
        X_train,
        y_train,
        title=f"Learning Curve - {model_name} (Evaluation)",
        scoring='f1',
        cv=5
    )

    # 5. Model's final evaluation with the ideal threshold
    ideal_th, best_f1, _, _, _ = get_optimal_threshold_from_scores(best_model, X_test, y_test)
    metrics_dict = full_classification_report_from_scores(
        best_model,
        X_test,
        y_test,
        threshold=ideal_th,
        model_name=model_name
    )

    print("Evaluation completed.")


if __name__ == "__main__":
     run_training_pipeline()

    # run_evaluation_pipeline()


