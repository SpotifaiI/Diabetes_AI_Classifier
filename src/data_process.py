import pandas as pd
import logging

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import KFold

logging.basicConfig(
    filename='./utils/app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def read_data(database_id):
    
    cdc_diabetes_health_indicators = fetch_ucirepo(id=database_id)
    x = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets.squeeze()
    logging.info(f"Data balance: {y.value_counts(normalize=True)}")
    print(x, y)
    
    return  x, y

def gen_kfold_and_save(x, filename, n_splits=3, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds_data = []
    for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
        folds_data.append({
            "fold": fold,
            "train_indices": list(train_index),
            "test_indices": list(test_index)
        })

    df_folds = pd.DataFrame(folds_data)
    df_folds.to_csv(filename, index=False)

    return df_folds

def load_kfold_df(filename):
    df_folds = pd.read_csv(filename)
    return df_folds