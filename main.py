from src.data_loader import load_stratified_data
from src.pipelines import get_preprocessing_pipeline
from src.train import (
    save_model,
    update_experiment_log,
    train_sgd_svm
    )
from src.evaluation import (
    analyze_feature_importance,
    get_optimal_threshold_from_scores,
    full_classification_report_from_scores
    )


def main():
    # Set the model name for saving and reporting
    model_name = "sgd_svm_v1"

    # 1. Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_stratified_data()

    # 2. Get preprocessing pipeline
    print("Setting up preprocessing pipeline...")
    preprocessor = get_preprocessing_pipeline()

    # 3. Model training and hyperparameter optimization (Lasso)
    print(f"Initiating training: {model_name}...")
    best_model, best_params = train_sgd_svm(X_train, y_train, preprocessor)
    print(f"Best hyperparameters: {best_params}")

    # 4. Significance analysis of variables
    analyze_feature_importance(best_model)

    # 5. Look for the ideal threshold
    ideal_th, best_f1, _, _, _ = get_optimal_threshold_from_scores(best_model, X_test, y_test)

    # 6. Model's final evaluation with the ideal threshold
    metrics_dict = full_classification_report_from_scores(
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

if __name__ == "__main__":
    main()