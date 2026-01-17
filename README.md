# Credit Scoring Model for Loan Approval
- Author: Facundo Mendez
- Objective: The primary goal of this project is to build a robust predictive system to determine whether a loan application should be approved or rejected. By automating the decision-making process, this model aims to manage credit risk efficiently and provide data-driven insights into applicant eligibility.

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
  4. **Model Selection**: Iterative testing of Linear Models (Logistic Regression, SGD) and Support Vector Machines (Linear, Poly, RBF).
  5. **Evaluation**: Focus on F1-Score (balance) and ROC-AUC, utilizing custom threshold tuning to optimize decision boundaries.

## 4. Project Architecture & Workflow
The project has been refactored into a modular structure to support MLOps best practices, separating training logic from evaluation and diagnostics.
- **src/data_loader.py**: Stratified data loading to handle class imbalance.
- **src/pipelines.py**: Scikit-Learn pipelines with custom transformers (FeatureCreator) and scalers.
- **src/train.pyn**: Logic for model training and hyperparameter optimization (GridSearchCV).
- **src/evaluation.py**: Advanced metrics, threshold tuning, and visual diagnostics (Learning Curves, Confusion Matrices, Support Vector Analysis).
- **main.py**: Entry point with dual execution modes:
    1. Training Pipeline: Trains new models from scratch, optimizes parameters, and logs results.
    2. Evaluation Pipeline: Loads saved models (.pkl) to perform deep diagnostics without re-training.

## 5. Key Techniques
  - **Optimization & Regularization**:
    - Lasso (L1): Used for automatic feature selection, effectively zeroing out redundant coefficients. 
    - SVM Kernels: Exploration of Linear, Polynomial, and RBF kernels to capture non-linear decision boundaries. 
  - **Threshold Tuning**: Instead of the default 0.5 threshold, the model applies a custom search to find the optimal threshold that maximizes the F1-Score. 
  - **Diagnostic Tools**:
    - Learning Curves: To detect overfitting/underfitting. 
    - Support Vector Analysis: Inspection of the "margin" density using PCA to understand dataset separability.

## 6. Experimental Results
The following table summarizes the performance across different architectures and optimization engines:

| Model                | ROC AUC | F1-Score | Precision | Recall | Adjusted Threshold |
|:---------------------|:-------:|:--------:|:---------:|:------:|:------------------:|
| **Logit Lasso (v7)** | 0.7466  |  0.8691  |  0.7830   | 0.9765 |       0.5201       |
| **SGD Logit (v1)**   | 0.7543  |  0.8691  |  0.7830   | 0.9765 |       0.5222       |
| **SGD SVM (v1)**     | 0.7418  |  0.8710  |  0.8020   | 0.9529 |       0.6799       |
| **Logit Poly (v2)**  | 0.7466  |  0.8691  |  0.7830   | 0.9760 |       0.5201       |
| **SVM_RBF_v1**       | 0.7325  |  0.8691  |  0.7830   | 0.9765 |       0.6538       |
| **SVM_Poly_v5**      | 0.7186  |  0.8691  |  0.7830   | 0.9765 |       0.4129       |

## Final Conclusion
This project successfully delivers a robust, production-ready pipeline for automated credit risk assessment. Through rigorous experimentation ranging from regularized linear models (Lasso, SGD) to non-linear Support Vector Machines (RBF, Polynomial kernels), the analysis conclusively demonstrated that the underlying data structure favors a parsimonious linear approach.
Key findings supporting this conclusion include:
  1. **Non-Linear Saturation**: Complex kernels did not outperform linear baselines. The RBF SVM required nearly 50% of the training data as Support Vectors, revealing a "noisy" margin with significant class overlap that geometric transformations could not resolve,.
  2. **Computational Efficiency**: The Polynomial SVM struggled to converge even after 150,000 iterations, introducing high computational cost without a gain in predictive power.
  Consequently, the Linear SVM (sgd_svm_v1) emerged as the optimal solution, achieving a stable F1-Score of 0.87 and a high Precision of 80.2%. This model effectively balances the trade-off between business opportunity and risk management, offering a transparent and mathematically stable tool for real-time loan eligibility prediction compared to "black-box" alternatives.
