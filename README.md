# Credit Scoring Model for Loan Approval
- **Author**: Facundo Mendez
- **Objective**: The primary goal of this project is to build a robust predictive system to determine whether a loan application should be approved or rejected. By automating the decision-making process, this model aims to manage credit risk efficiently and provide data-driven insights into applicant eligibility.

## 1. Data Dictionary
The dataset consists of the following primary features as described in the initial project documentation:
- **Gender**: Gender of the applicant ("Male" or "Female").
- **Married**: Marital status of the applicant ("Yes" or "No").
- **Dependents**: Number of people economically dependent on the applicant (e.g., 0, 1, 2, "3+").
- **Education**: Applicant's educational level ("Graduate" or "Not Graduate").
- **Self_Employed**: Whether the applicant is self-employed ("Yes" or "No").
- **ApplicantIncome**: Annual or monthly income of the applicant.
- **CoapplicantIncome**: Annual or monthly income of the co-applicant.
- **LoanAmount**: Total loan amount requested.
- **Loan_Amount_Term**: Maturity term of the loan.
- **Credit_History**: Credit history indicator (1 for good, 0 for problematic).
- **Property_Area**: Location of the property ("Urban", "Rural", or "Semiurban").
- **loan_status**: The target variable (Output), indicating if the loan was approved (Y) or not (N).

## 2. Exploratory Data Analysis (EDA) Highlights
The initial data exploration revealed critical patterns for modeling:
- **Key Predictor**: Credit_History is the most dominant factor in loan approval; applicants with a positive history have a significantly higher approval rate. 
- **Class Imbalance**: Approximately 68.6% of the loans in the training set are approved, which necessitates the use of metrics like F1-Score rather than just Accuracy.
- **Feature Distributions**: Numerical variables exhibited high skewness, justifying the use of robust scaling (MinMaxScaler) within the preprocessing pipeline.

## 3. Methodology
The project follows a rigorous Machine Learning workflow:
  1. **Data Loading**: Stratified splitting to ensure representative distribution of the target variable (Loan_Status).
  2. **Preprocessing**: Automated pipelines for missing value imputation, scaling (MinMaxScaler), and encoding (OneHotEncoder).
  3. **Feature Engineering**: Creation of financial ratios (e.g., Income_Loan_Ratio) to enhance predictive power.
  4. **Model Selection**:
     - **Phase 1, 2 & 3**: Linear Models (Logistic Regression, SGD), Support Vector Machines (Linear, Poly, RBF) and Decision Trees.
     - **Phase 4 (Ensembles)**: Implementation of advanced tree-based methods including Random Forest, Extra Trees, and Boosting algorithms (AdaBoost, Gradient Boosting, XGBoost, LightGBM). 
     - **Phase 5 (Stacking)**: Construction of a Stacking Ensemble to combine the strengths of heterogeneous models.
  5. **Evaluation**: Focus on F1-Score (balance) and ROC-AUC, utilizing custom threshold tuning to optimize decision boundaries.

## 4. Project Architecture & Workflow
The project has been refactored into a modular structure to support MLOps best practices, separating training logic from evaluation and diagnostics.
- **src/data_loader.py**: Stratified data loading to handle class imbalance.
- **src/pipelines.py**: Scikit-Learn pipelines with custom transformers (FeatureCreator) and scalers.
- **src/train.pyn**: Logic for model training, including hyperparameter optimization (GridSearchCV and RandomizedSearchCV).
- **src/evaluation.py**: Advanced metrics, threshold tuning, and visual diagnostics (Learning Curves, Confusion Matrices, Support Vector Analysis, Tree Visualization).
- **main.py**: Entry point with dual execution modes:
    1. Training Pipeline: Trains new models from scratch, optimizes parameters, and logs results.
    2. Evaluation Pipeline: Loads saved models (.pkl) to perform deep diagnostics without re-training.

## 5. Key Techniques
  - **Optimization & Regularization**:
    - **Lasso (L1)**: Used for automatic feature selection, effectively zeroing out redundant coefficients to reduce noise. 
    - **SVM Kernels**: Exploration of Linear, Polynomial, and RBF kernels to capture non-linear decision boundaries.
    - **Tree Pruning**: Usage of `max_depth` and `min_samples_leaf` to control variance and prevent overfitting in Decision Trees.
    - **Boosting Regularization**: Usage of reg_alpha (L1) and reg_lambda (L2) in XGBoost/LightGBM to control overfitting on the small dataset.
  - **Ensemble Strategies**:
    - **Bagging**: Reduction of variance using Random Forests and Extra Trees. 
    - **Boosting**: Sequential correction of errors using Gradient Boosting, XGBoost, and LightGBM. Special attention was paid to class imbalance using scale_pos_weight and is_unbalance parameters. 
    - **Stacking**: Using a Logistic Regression meta-learner to combine predictions from Extra Trees, Random Forest, XGBoost, LightGBM, and a base Logistic Regression model. This successfully combined the stability of linear models with the non-linear capabilities of tree-based estimators.
  - **Threshold Tuning**: Instead of the default 0.5 threshold, the model applies a custom search to find the optimal threshold that maximizes the F1-Score. 
  - **Diagnostic Tools**:
    - **Learning Curves**: To detect overfitting/underfitting. 
    - **Support Vector Analysis**: Inspection of the "margin" density using PCA.
    - **Confusion Matrices**: Visual representation of classification performance.
    - **Tree Visualization**: Exporting decision logic to Graphviz (.dot/.png) for interpretability.
    - **Meta-Learner Weight Analysis**: Inspection of the stacking meta-model coefficients to understand the contribution of each base estimator.

## 6. Experimental Results
The following table summarizes the performance across different architectures and optimization engines:

| Model                      | ROC AUC | F1-Score | Precision | Recall | Adjusted Threshold |
|:---------------------------|:-------:|:--------:|:---------:|:------:|:------------------:|
| **Logit Lasso (v7)**       | 0.7466  |  0.8691  |  0.7830   | 0.9765 |       0.5201       |
| **SGD Logit (v1)**         | 0.7543  |  0.8691  |  0.7830   | 0.9765 |       0.5222       |
| **SGD SVM (v1)**           | 0.7418  |  0.8710  |  0.8020   | 0.9529 |       0.6799       |
| **Logit Poly (v2)**        | 0.7466  |  0.8691  |  0.7830   | 0.9760 |       0.5201       |
| **SVM_RBF_v1**             | 0.7325  |  0.8691  |  0.7830   | 0.9765 |       0.6538       |
| **SVM_Poly_v5**            | 0.7186  |  0.8691  |  0.7830   | 0.9765 |       0.4129       |
| **Decision Tree (v1)**     | 0.7269  |  0.8600  |  0.7800   | 0.9600 |       0.7569       |
| **Random Forest (v1)**     | 0.7218  |  0.8737  |  0.7905   | 0.9765 |       0.4661       |
| **Extra Trees (v1)**       | 0.7745  |  0.8691  |  0.7830   | 0.9765 |       0.6189       |
| **AdaBoost (v1)**          | 0.7404  |  0.8691  |  0.7830   | 0.9765 |       0.4062       |
| **Gradient Boosting (v2)** | 0.6952  |  0.8705  |  0.7778   | 0.9882 |       0.1839       |
| **LightGBM (v1)**          | 0.7328  |  0.8737  |  0.7905   | 0.9765 |       0.4974       |
| **XGBoost (v3)**           | 0.7314  |  0.8691  |  0.7830   | 0.9765 |       0.5000       |
| **Stacking (v2)**          | 0.8037  |  0.8791  |  0.8247   | 0.9412 |       0.6678       |

## Final Conclusion
This project demonstrates the evolution from baseline linear models to state-of-the-art ensemble techniques. While earlier phases suggested the problem was predominantly linear, the introduction of Stacking provided the definitive breakthrough.

**Key Findings**:
1. **The "Linear Ceiling"**: Individual models (both linear and tree-based) struggled to surpass an ROC-AUC of ~0.75. Models like XGBoost and Random Forest, while powerful, did not individually outperform the simple Logistic Regression significantly on this small dataset.
2. **The Stacking Solution**: The Stacking v2 model successfully broke the performance ceiling, achieving an ROC-AUC of 0.8037. By combining a Logistic Regression (as a meta-learner) with Extra Trees, Random Forest, XGBoost, and LightGBM, the model learned to leverage the stability of linear patterns while capturing subtle non-linearities via the tree-based estimators.
3. **Business Impact**: The Stacking model optimized with a threshold of 0.67 achieved a Precision of 82.5% with a Recall of 94%. This configuration minimizes the financial risk of approving bad loans (False Positives) without alienating valid customers (maintaining high True Positives).

The final pipeline represents a production-ready solution that balances complexity and performance, utilizing the best of both linear interpretability and ensemble robustness.