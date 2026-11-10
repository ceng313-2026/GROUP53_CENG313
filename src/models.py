#models


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb

CAT_COLS = ['country', 'store', 'product']
NUM_COLS = ['month', 'dayofweek', 'dayofyear', 'is_weekend']

#baseline 2

def train_ridge(train_df, val_df, alpha=1.0):
    #Trains the Ridge model using a log1p-transformed target and converts validation predictions back to the original scale.
    X_tr = pd.get_dummies(train_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
    X_va = pd.get_dummies(val_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

    y_tr = np.log1p(train_df['num_sold'])

    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    return np.expm1(model.predict(X_va))

#baseline 3 

def train_lgbm(train_df, val_df, n_estimators=500, learning_rate=0.05):
    """LightGBM'i log1p hedefiyle eğitir, validation tahminlerini (orijinal ölçekte) döner."""
    X_tr = train_df[CAT_COLS + NUM_COLS].copy()
    X_va = val_df[CAT_COLS + NUM_COLS].copy()
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype('category')
        X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

    y_tr = np.log1p(train_df['num_sold'])

    model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, verbose=-1)
    model.fit(X_tr, y_tr)
    return np.expm1(model.predict(X_va))