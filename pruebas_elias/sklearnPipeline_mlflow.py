#%%
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
# from tpot import TPOTClassifier
import itertools

def confusion_matrix_to_dataframe(confusion_matrix):
    """
    Convert a confusion matrix to a DataFrame with specific columns and rows.
    
    Parameters:
    confusion_matrix (numpy.ndarray): A 2x2 confusion matrix.
    
    Returns:
    pandas.DataFrame: A DataFrame with columns 'Predicted Positive', 'Predicted Negative' 
                      and rows 'Actual Positive', 'Actual Negative'.
    """
    # Ensure the input is a 2x2 numpy array
    if confusion_matrix.shape != (2, 2):
        raise ValueError("Confusion matrix must be a 2x2 matrix")
    
    # Create the DataFrame
    df = pd.DataFrame(confusion_matrix, 
                      columns=['Predicted Negative', 'Predicted Positive'], 
                      index=['Actual Negative', 'Actual Positive'])
    
    return df
#%% Reading data

base = pd.read_csv('data_chida_diego.csv').drop(columns=['date'])

#Dividing training-testing
base = base.sort_values(by=['year', 'month'], ascending=[True, True])

train_val_df = base[(base['year'] < 2023) | ((base['year'] == 2023) & (base['month'] < 5))]
test_df = base[(base['year'] > 2023) | ((base['year'] == 2023) & (base['month'] >= 5))]

base = None
#%%
# Split the balanced dataframe into train and validation sets
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

#* Balancing training
# Select rows with "churn_next_month" value of 1
churn_1 = train_df[train_df["churn_next_month"] == 1]

# Randomly sample the same amount of rows with "churn_next_month" value of 0
churn_0 = train_df[train_df["churn_next_month"] == 0].sample(n=len(churn_1), random_state=42)

# Concatenate the sampled dataframes
train_df = pd.concat([churn_1, churn_0])



#%%

cat_cols = ['type', 
            'cluster'
            ]

num_cols = ['month', 
            'amount', 
            'year',
            'antiguedad', 
            'componente_estacional', 
            'varianza',
            'promedio_temporada', 
            'porcentaje_vs_promedio',
            'tiempo_desde_ultima_compra', 
            'percentage_change'
            ]

id_cols = ['customer_id']

target = 'churn_next_month'

#%% Defining things to run over
X_train, y_train = train_df[cat_cols + num_cols], train_df[target]
X_val, y_val = val_df[cat_cols + num_cols], val_df[target]
X_test, y_test = test_df[cat_cols + num_cols], test_df[target]

#%%Defining grid search

param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.03, 0.1, 0.015, 0.06],
    'max_depth': [7, 13, 20],
    # 'min_child_weight': [1, 5, 10],
    # 'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.25, 0.6, 0.8],
    'scale_pos_weight':[1/3, 1/7, 1/10, 1/14, 1/20],
    'reg_lambda':[0,0.5,1,2],
    # 'reg_alpha':[0,0.5,1,2],
    # 'sampling_method':['uniform', 'gradient_based']
}
# param_grid = {
#     'n_estimators': [100],
#     'learning_rate': [0.1, 0.2],
#     'max_depth': [20],
#     # 'min_child_weight': [1, 5, 10],
#     # 'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.25, 0.6],
#     'scale_pos_weight':[1/7, 1/10, 1/14, 1/20],

# }

param_combinations = list(itertools.product(
    param_grid['n_estimators'],
    param_grid['learning_rate'],
    param_grid['max_depth'],
    # param_grid['min_child_weight'],
    # param_grid['subsample'],
    param_grid['colsample_bytree'],
    param_grid['scale_pos_weight'],
    param_grid['reg_lambda']
    # param_grid['reg_alpha'],
    # param_grid['sampling_method']
))

best_score = 0
best_args = None
best_model = None
tot = len(param_combinations)
i=0

for (n_estimators, learning_rate, max_depth, colsample_bytree, scale_pos_weight, reg_lambda) in param_combinations:
    
    i += 1
    print(f'--------------------------------------Run: {i} of {tot}---------------------------------------------')
    print((n_estimators, learning_rate, max_depth, colsample_bytree, scale_pos_weight, reg_lambda))
    print()
    with mlflow.start_run():
        
        mlflow.log_param('regressor', XGBClassifier(
                                        objective='binary:logistic',
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        colsample_bytree=colsample_bytree,
                                        seed=42,
                                        scale_pos_weight=scale_pos_weight,
                                        # reg_alpha=reg_alpha,
                                        reg_lambda=reg_lambda
                                        ))
        
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('colsample_bytree', colsample_bytree)
        mlflow.log_param('scale_pos_weight', scale_pos_weight)
        # mlflow.log_param('reg_alpha', reg_alpha)
        mlflow.log_param('reg_lambda', reg_lambda)
        
        
        categorical_preprocessor = Pipeline(steps=[('encoder', OneHotEncoder())])

        numerical_preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

        preprocessor = ColumnTransformer(transformers=[
            ('categorical', categorical_preprocessor, cat_cols),
            ('numerical', numerical_preprocessor, num_cols),
            
        ])


        model = Pipeline(steps=[('pre', preprocessor),
                                ('classifier', XGBClassifier(
                                        objective='binary:logistic',
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        colsample_bytree=colsample_bytree,
                                        seed=42,
                                        scale_pos_weight=scale_pos_weight,
                                        # reg_alpha=reg_alpha,
                                        reg_lambda=reg_lambda
                                        )
                                )]
                        )

        model.fit(X_train, y_train)

        # Train Results

        # Predecir los resultados para el conjunto de validación
        y_pred_train = model.predict(X_train)

        acc_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)
        roc_train = roc_auc_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)
        
        mlflow.log_metric("accuracy_train", acc_train)
        mlflow.log_metric("precision_train", precision_train)
        mlflow.log_metric("roc_train", roc_train)
        mlflow.log_metric("f1_score_train", f1_train)
        mlflow.sklearn.log_model(model, "best_model")
        
        # Val Results

        # Predecir los resultados para el conjunto de validación
        y_pred_val = model.predict(X_val)

        acc_val = accuracy_score(y_val, y_pred_val)
        precision_val = precision_score(y_val, y_pred_val)
        roc_val = roc_auc_score(y_val, y_pred_val)
        f1_val = f1_score(y_val, y_pred_val)
        
        mlflow.log_metric("accuracy_val", acc_val)
        mlflow.log_metric("precision_val", precision_val)
        mlflow.log_metric("roc_val", roc_val)
        mlflow.log_metric("f1_score_val", f1_val)
        
        # Test Results

        # Predecir los resultados para el conjunto de validación
        y_pred_test = model.predict(X_test)

        acc_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        roc_test = roc_auc_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)
        
        mlflow.log_metric("accuracy_test", acc_test)
        mlflow.log_metric("precision_test", precision_test)
        mlflow.log_metric("roc_test", roc_test)
        mlflow.log_metric("f1_score_test", f1_test)
