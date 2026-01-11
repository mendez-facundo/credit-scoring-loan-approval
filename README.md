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

## 3. Methodology and Modular Pipeline
To ensure scalability and reproducibility, the project implements a modular architecture using Scikit-Learn's Pipeline and ColumnTransformer:
- **Feature Engineering**: Custom transformers were developed to create new features such as Total_Income, Income_Loan_Ratio, and High_Income_Flag.
- **Automated Selection (Lasso)**: L1 regularization was applied to perform automatic feature selection, effectively reducing redundant coefficients (such as High_Income_Flag) to zero.
- **Optimization**: Utilized GridSearchCV to fine-tune regularization strength and SGDClassifier for efficient stochastic optimization.
- **Threshold Tuning**: Instead of the default 0.5 threshold, the model applies a custom search to find the optimal threshold that maximizes the F1-Score, balancing Precision and Recall.

## 4. Experimental Results
The following table summarizes the performance across different architectures and optimization engines:

| Model | ROC AUC | F1-Score | Precision | Recall | Adjusted Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logit Lasso (v7)** | 0.7466 | 0.8691 | 0.7830 | 0.9765 | 0.5201 |
| **SGD Logit (v1)** | 0.7543 | 0.8691 | 0.7830 | 0.9765 | 0.5222 |
| **SGD SVM (v1)** | 0.7418 | 0.8710 | 0.8020 | 0.9529 | 0.6799 |

## Final Conclusion
The sgd_svm_v1 (Linear SVM) model achieved the highest Precision (80.2%), effectively reducing the number of risky loans approved (False Positives) while maintaining a strong overall F1-Score.
