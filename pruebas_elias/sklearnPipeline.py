#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from tpot import TPOTClassifier

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

#%% Defining processors

categorical_preprocessor = Pipeline(steps=[('encoder', OneHotEncoder())])

numerical_preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[
    ('categorical', categorical_preprocessor, cat_cols),
    ('numerical', numerical_preprocessor, num_cols),
    
])

args = dict()
args['n_estimators'] = 100
args['learning_rate'] = 0.03
args['max_depth'] = 13
args['colsample_bytree'] = 0.6
args['scale_pos_weight'] = 0.05

# model = Pipeline(steps=[('pre', preprocessor),
#                         ('classifier', XGBClassifier(
#                                 objective='binary:logistic',
#                                 n_estimators=args['n_estimators'],
#                                 learning_rate=args['learning_rate'],
#                                 max_depth=args['max_depth'],
#                                 colsample_bytree=args['colsample_bytree'],
#                                 seed=42,
#                                 scale_pos_weight=args['scale_pos_weight']
#                                 )
#                          )]
#                  )
model = Pipeline(steps=[('pre', preprocessor),
                        ('classifier', TPOTClassifier(verbosity=2, 
                                                      generations=8, 
                                                      population_size=20, 
                                                      random_state=42)
                         )]
                 )

model.fit(X_train, y_train)

#%% Train Results

# Predecir los resultados para el conjunto de validación
y_pred_train = model.predict(X_train)

# Calcular y mostrar el accuracy y el informe de clasificación
accuracy = accuracy_score(y_train, y_pred_train)
conf_matrix = confusion_matrix(y_train, y_pred_train)
class_report = classification_report(y_train, y_pred_train)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix_to_dataframe(conf_matrix))
print("Classification Report:\n", class_report)

#%% Validation Results

# Predecir los resultados para el conjunto de validación
y_pred_val = model.predict(X_val)

# Calcular y mostrar el accuracy y el informe de clasificación
accuracy = accuracy_score(y_val, y_pred_val)
conf_matrix = confusion_matrix(y_val, y_pred_val)
class_report = classification_report(y_val, y_pred_val)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix_to_dataframe(conf_matrix))
print("Classification Report:\n", class_report)

#%%Test Results

# Predecir los resultados para el conjunto de validación
y_pred_test = model.predict(X_test)

# Calcular y mostrar el accuracy y el informe de clasificación
accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)
class_report = classification_report(y_test, y_pred_test)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix_to_dataframe(conf_matrix))
print("Classification Report:\n", class_report)

# %%
