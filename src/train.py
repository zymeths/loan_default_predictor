"""
train.py
--------
Stratified K-Fold training pipeline for the loan default predictor.
Target encoding is applied inside each fold to prevent leakage.
Produces out-of-fold predictions and averaged test predictions.

Usage:
    python src/train.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier

from features import encode_base, engineer_features, DROP_COLS, TARGET_ENCODE_COLS

# ── Paths
TRAIN_PATH  = 'data/train.csv'
TEST_PATH   = 'data/test.csv'
SUBMIT_PATH = 'submissions/submission_v2.csv'
TARGET_COL  = 'loan_paid_back'

# ── Best params from Optuna tuning
BEST_PARAMS = {
    'n_estimators':      929,
    'max_depth':         3,
    'learning_rate':     0.20500355271576148,
    'num_leaves':        136,
    'min_child_samples': 55,
    'subsample':         0.970525603747719,
    'colsample_bytree':  0.730946592537135,
    'reg_alpha':         0.04488386265135877,
    'reg_lambda':        0.028378759246160797,
    'class_weight':      'balanced',
    'random_state':      42,
    'verbose':           -1,
}

N_SPLITS = 5


def run_training():
    # ── Load
    print('Loading data...')
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    # ── Base encoding (ordinal + label — no target, safe outside folds)
    df_train, df_test = encode_base(df_train, df_test)

    X_raw      = df_train.drop(columns=['id', TARGET_COL])
    y          = df_train[TARGET_COL]
    X_test_raw = df_test.drop(columns=['id'])

    oof_preds  = np.zeros(len(X_raw))
    test_preds = np.zeros(len(X_test_raw))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        print(f'\nFold {fold + 1}/{N_SPLITS}')

        X_tr  = X_raw.iloc[train_idx].copy()
        X_val = X_raw.iloc[val_idx].copy()
        X_te  = X_test_raw.copy()
        y_tr  = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # ── Target encoding inside fold (prevents leakage)
        te = TargetEncoder(cols=TARGET_ENCODE_COLS, smoothing=10)
        X_tr[TARGET_ENCODE_COLS]  = te.fit_transform(X_tr[TARGET_ENCODE_COLS], y_tr)
        X_val[TARGET_ENCODE_COLS] = te.transform(X_val[TARGET_ENCODE_COLS])
        X_te[TARGET_ENCODE_COLS]  = te.transform(X_te[TARGET_ENCODE_COLS])

        # ── Feature engineering inside fold
        X_tr  = engineer_features(X_tr).drop(columns=DROP_COLS)
        X_val = engineer_features(X_val).drop(columns=DROP_COLS)
        X_te  = engineer_features(X_te).drop(columns=DROP_COLS)

        # ── Train
        model = LGBMClassifier(**BEST_PARAMS)
        model.fit(X_tr, y_tr)

        # ── Validate
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f'  AUC: {fold_auc:.4f}')

        # ── Test predictions averaged across folds
        test_preds += model.predict_proba(X_te)[:, 1] / N_SPLITS

    print(f'\nOverall OOF AUC: {roc_auc_score(y, oof_preds):.4f}')

    # ── Save submission
    submission = pd.DataFrame({
        'id':             df_test['id'],
        TARGET_COL:       test_preds
    })
    submission.to_csv(SUBMIT_PATH, index=False)
    print(f'Submission saved to {SUBMIT_PATH}')


if __name__ == '__main__':
    run_training()
