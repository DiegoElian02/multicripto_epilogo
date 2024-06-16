import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, fbeta_score

def compute_f_score(df: pd.DataFrame, beta: float = 1):
    f_score = fbeta_score(df["churn"], df["churn_pred"], beta=beta)
    return f_score

