import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import KernelPCA
from sklearn import set_config
from src.transformers import ColumnDropper, FeatureCreator

set_config(transform_output="pandas")

def get_preprocessing_pipeline():
    # 1. Set numerical and categorical features
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                          'Loan_Amount_Term', 'Credit_History']
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education',
                            'Self_Employed', 'Property_Area']

    # 2. Replacing missing values with global imputers
    global_imputer = ColumnTransformer(transformers=[
        ('num_imputer', SimpleImputer(strategy='mean'), numerical_features),
        ('cat_impute', SimpleImputer(strategy='most_frequent'), categorical_features)
    ], remainder='passthrough', verbose_feature_names_out=False)

    # 3. Consolidated dynamic pipeline for both types of features
    final_processing = ColumnTransformer(transformers=[
        ('num_scaler', MinMaxScaler(), make_column_selector(dtype_include=np.number)),
        ('cat_encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
         make_column_selector(dtype_include=object))
    ], remainder='passthrough', verbose_feature_names_out=False)

    # 4. Main pipeline
    full_pipeline = Pipeline(steps=[
        ('initial_dropper', ColumnDropper(columns_to_drop=['Loan_ID'])),
        ('imputer', global_imputer),
        ('feature_creator', FeatureCreator()),
        ('preprocessor', final_processing)
    ])

    return full_pipeline


def get_kpca_pipeline(n_components=10, kernel='rbf', gamma=0.04):
    """
    Extend the standard preprocessing pipeline by adding Kernel PCA at the end.
    """
    # Import the base preprocessing pipeline
    base_pipeline = get_preprocessing_pipeline()

    # Incorporate Kernel PCA to the pipeline
    kpca_pipeline = Pipeline(steps=[
        ('base_preprocessor', base_pipeline),
        ('kpca', KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, fit_inverse_transform=False))
    ])

    return kpca_pipeline