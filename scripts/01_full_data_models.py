#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Primary manuscript model: RF9, RF25, Weighted XGB25, weighted fusion, Platt scaling.
Curated from IJMS_publication_code_model.ipynb code cells 2,4,6,8,10,12,14.
No original private clinical records or model artifacts are included.
Full computation is expensive; this script has not been validated against private data here.
"""
import argparse
import os
import json
import warnings
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

PARSER=argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--data-path', type=Path, default=Path(os.getenv('CHD_DATA_PATH', 'data/private/allmax_sort.csv')))
PARSER.add_argument('--output-dir', type=Path, default=Path('outputs/full_data'))
PARSER.add_argument('--jobs', type=int, default=8, help='Parallel inner grid search workers')
PARSER.add_argument('--save-models', action='store_true', help='Save trained models locally for authorized prediction use; do not publish fitted artifacts')
ARGS=PARSER.parse_args()
if not ARGS.data_path.is_file():
    PARSER.error('Private, cleaned X25 CSV not found. Pass --data-path /secure/path/allmax_sort.csv.')
ARGS.output_dir.mkdir(parents=True,exist_ok=True)
# ===== PUBLICATION-SOURCE CELL 2 =====
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

# Private input; change the local path only if running this notebook in an authorized environment.
DATA_PATH = ARGS.data_path
TARGET = 'ill_CHD'
SPLIT_SEED = 42
GLOBAL_SEED = 20261009
FEATURES_9 = [
    'Age', 'WHtR', 'BMI', 'ill_Stroke', 'Smoking',
    'ill_Chronic_Lung_Disease', 'Sex', 'Drinking', 'MSP'
]
FEATURES_25 = FEATURES_9 + [
    'SC', 'Heart_Murmur', 'FBG', 'DPD', 'SPD', 'BUN', 'PLT', 'Hb',
    'Barrel_Chest', 'HDL', 'LDL', 'SOB', 'Hepatitis_B_surface_antigen',
    'WBC', 'TC/HDL', 'TG'
]
assert len(FEATURES_9) == 9 and len(FEATURES_25) == 25

data = pd.read_csv(DATA_PATH, encoding='GBK')
non_chd, chd = data[data[TARGET] == 0], data[data[TARGET] == 1]
non_chd_dev, non_chd_test = train_test_split(non_chd, test_size=0.10, random_state=SPLIT_SEED)
chd_dev, chd_test = train_test_split(chd, test_size=0.10, random_state=SPLIT_SEED)
train_data = pd.concat([non_chd_dev, chd_dev], axis=0)
test_data = pd.concat([non_chd_test, chd_test], axis=0)

X_dev, X_test = train_data[FEATURES_25].copy(), test_data[FEATURES_25].copy()
y_dev, y_test = train_data[TARGET].copy(), test_data[TARGET].copy()
assert (len(X_dev), len(X_test)) == (108298, 12034)
assert tuple(np.bincount(y_dev.to_numpy())) == (104840, 3458)
assert tuple(np.bincount(y_test.to_numpy())) == (11649, 385)
oof_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_dev, y_dev))

# Row-level separation does not establish patient-level independence.

assert not data.columns.duplicated().any(), 'Duplicate feature names in cleaned input.'
assert data[TARGET].notna().all() and set(data[TARGET].unique())=={0,1}
assert all(col in data.columns for col in FEATURES_25+[TARGET]), 'Missing expected 25 predictors'
assert len(data)==120332 and int(data[TARGET].sum())==3843, 'Input does not match manuscript cohort.'
assert not X_dev.isna().any().any() and not X_test.isna().any().any(), 'Private input must be pre-cleaned.'


# ===== PUBLICATION-SOURCE CELL 4 =====
# Source Step 63: the second assignment replaced the earlier broader search grid.
RF9_OOF_GRID = {
    'n_estimators': [600], 'criterion': ['gini', 'entropy'],
    'max_depth': [10, 15, 20], 'min_samples_split': [5, 10, 20],
    'max_leaf_nodes': [100]
}
published_rf_oof_prob = np.full(len(y_dev), np.nan)
published_rf_fold_results = []
for fold_id, (fit_idx, val_idx) in enumerate(oof_splits, start=1):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=3000 + fold_id)
    rf = RandomForestClassifier(random_state=4000 + fold_id, n_jobs=1, class_weight=None)
    search = GridSearchCV(rf, RF9_OOF_GRID, cv=inner_cv, scoring='roc_auc',
                          n_jobs=ARGS.jobs, refit=True, error_score='raise')
    search.fit(X_dev.iloc[fit_idx][FEATURES_9], y_dev.iloc[fit_idx])
    prob = search.best_estimator_.predict_proba(X_dev.iloc[val_idx][FEATURES_9])[:, 1]
    published_rf_oof_prob[val_idx] = prob
    published_rf_fold_results.append({'fold': fold_id, 'best_params': search.best_params_,
                                      'inner_auc': search.best_score_,
                                      'outer_auc': roc_auc_score(y_dev.iloc[val_idx], prob),
                                      'outer_pr_auc': average_precision_score(y_dev.iloc[val_idx], prob)})
assert np.isfinite(published_rf_oof_prob).all()

# Source Step 82: RF9 final training used 5-fold inner CV with seed 42.
RF_FINAL_GRID = {'criterion': ['gini', 'entropy'], 'max_depth': [15, 20],
                 'min_samples_split': [5, 10, 20]}
rf2024_search = GridSearchCV(
    RandomForestClassifier(n_estimators=600, max_leaf_nodes=100, random_state=42, n_jobs=1),
    RF_FINAL_GRID,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=ARGS.jobs, refit=True, error_score='raise')
rf2024_search.fit(X_dev[FEATURES_9], y_dev)
final_rf2024 = rf2024_search.best_estimator_
rf2024_test_prob = final_rf2024.predict_proba(X_test[FEATURES_9])[:, 1]


# ===== PUBLICATION-SOURCE CELL 6 =====
RF25_PARAM_GRID = {
    'criterion': ['gini', 'entropy'], 'max_depth': [15, 20],
    'min_samples_split': [5, 10, 20]
}
rf25_oof_prob = np.full(len(y_dev), np.nan)
rf25_fold_results = []
for fold_id, (train_idx, val_idx) in enumerate(oof_splits, start=1):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True,
                               random_state=GLOBAL_SEED + fold_id * 100)
    model = RandomForestClassifier(n_estimators=600, max_leaf_nodes=100,
                                   random_state=GLOBAL_SEED + fold_id * 1000, n_jobs=1)
    search = GridSearchCV(model, RF25_PARAM_GRID, cv=inner_cv, scoring='roc_auc',
                          n_jobs=ARGS.jobs, refit=True, error_score='raise')
    search.fit(X_dev.iloc[train_idx][FEATURES_25], y_dev.iloc[train_idx])
    prob = search.best_estimator_.predict_proba(X_dev.iloc[val_idx][FEATURES_25])[:, 1]
    rf25_oof_prob[val_idx] = prob
    rf25_fold_results.append({'fold': fold_id, 'best_params': search.best_params_,
                              'outer_auc': roc_auc_score(y_dev.iloc[val_idx], prob),
                              'outer_pr_auc': average_precision_score(y_dev.iloc[val_idx], prob)})
assert np.isfinite(rf25_oof_prob).all()

final_rf25_search = GridSearchCV(
    RandomForestClassifier(n_estimators=600, max_leaf_nodes=100,
                           random_state=GLOBAL_SEED + 9999, n_jobs=1),
    RF25_PARAM_GRID,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=GLOBAL_SEED + 999),
    scoring='roc_auc', n_jobs=ARGS.jobs, refit=True, error_score='raise')
final_rf25_search.fit(X_dev[FEATURES_25], y_dev)
final_rf25_model = final_rf25_search.best_estimator_
rf25_test_prob = final_rf25_model.predict_proba(X_test[FEATURES_25])[:, 1]


# ===== PUBLICATION-SOURCE CELL 8 =====
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
    'learning_rate': 0.02, 'n_estimators': 400, 'max_depth': 4,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0,
    'reg_alpha': 0.0, 'scale_pos_weight': 3.0,
    'random_state': GLOBAL_SEED, 'n_jobs': ARGS.jobs
}
xgb25_oof_prob = np.full(len(y_dev), np.nan)
xgb25_fold_results = []
for fold_id, (train_idx, val_idx) in enumerate(oof_splits, start=1):
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_dev.iloc[train_idx][FEATURES_25], y_dev.iloc[train_idx])
    prob = model.predict_proba(X_dev.iloc[val_idx][FEATURES_25])[:, 1]
    xgb25_oof_prob[val_idx] = prob
    xgb25_fold_results.append({'fold': fold_id,
                               'ROC_AUC': roc_auc_score(y_dev.iloc[val_idx], prob),
                               'PR_AUC': average_precision_score(y_dev.iloc[val_idx], prob)})
assert np.isfinite(xgb25_oof_prob).all()
final_xgb25 = XGBClassifier(**XGB_PARAMS)
final_xgb25.fit(X_dev[FEATURES_25], y_dev)
xgb25_test_prob = final_xgb25.predict_proba(X_test[FEATURES_25])[:, 1]


# ===== PUBLICATION-SOURCE CELL 10 =====
# Development-only cross-fitted selection over RF9/XGB25 branch weights.
# AUC is primary; average precision resolves exact/near ties in the source selection.
fold_labels = np.zeros(len(y_dev), dtype=int)
for fold_id, (_, val_idx) in enumerate(oof_splits, start=1):
    fold_labels[val_idx] = fold_id
weighted_cf_prob = np.full(len(y_dev), np.nan)
weight_results = []
for held_fold in range(1, 6):
    train_mask, val_mask = fold_labels != held_fold, fold_labels == held_fold
    ranked = []
    for w_rf in np.arange(0.0, 1.0001, 0.02):
        p = w_rf * published_rf_oof_prob[train_mask] + (1-w_rf) * xgb25_oof_prob[train_mask]
        ranked.append((roc_auc_score(y_dev.to_numpy()[train_mask], p),
                       average_precision_score(y_dev.to_numpy()[train_mask], p), float(w_rf)))
    best_auc, best_ap, best_rf_weight = max(ranked, key=lambda x: (x[0], x[1]))
    weighted_cf_prob[val_mask] = (best_rf_weight * published_rf_oof_prob[val_mask]
                                  + (1-best_rf_weight) * xgb25_oof_prob[val_mask])
    weight_results.append({'fold': held_fold, 'rf_weight': best_rf_weight,
                           'xgb_weight': 1 - best_rf_weight})

# The manuscript's fixed final two-branch weights (selected in the original development analysis).
LOCKED_RF_WEIGHT, LOCKED_XGB_WEIGHT = 0.68, 0.32
raw_dev_prob = LOCKED_RF_WEIGHT * published_rf_oof_prob + LOCKED_XGB_WEIGHT * xgb25_oof_prob
raw_test_prob = LOCKED_RF_WEIGHT * rf2024_test_prob + LOCKED_XGB_WEIGHT * xgb25_test_prob


# ===== PUBLICATION-SOURCE CELL 12 =====
calibrated_dev_cf = np.full(len(y_dev), np.nan)
calibration_fold_results = []
for fold_id, (train_idx, val_idx) in enumerate(oof_splits, start=1):
    platt_fold = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
    platt_fold.fit(raw_dev_prob[train_idx].reshape(-1, 1), y_dev.iloc[train_idx])
    calibrated_dev_cf[val_idx] = platt_fold.predict_proba(
        raw_dev_prob[val_idx].reshape(-1, 1))[:, 1]
    calibration_fold_results.append({'fold': fold_id,
                                     'intercept': float(platt_fold.intercept_[0]),
                                     'coefficient': float(platt_fold.coef_[0, 0])})
final_platt = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
final_platt.fit(raw_dev_prob.reshape(-1, 1), y_dev)
calibrated_test_prob = final_platt.predict_proba(raw_test_prob.reshape(-1, 1))[:, 1]


# ===== PUBLICATION-SOURCE CELL 14 =====
# Optional scalar checks only. This cell does not create publication figures or tables.
def evaluation_summary(model_name, y, prob):
    return {'model': model_name,
            'ROC_AUC': roc_auc_score(y, prob),
            'PR_AUC': average_precision_score(y, prob),
            'Brier': brier_score_loss(y, prob),
            'LogLoss': log_loss(y, prob)}

model_score_arrays = {
    'RF9': rf2024_test_prob,
    'RF25': rf25_test_prob,
    'Weighted_XGB25': xgb25_test_prob,
    'Raw_Ensemble': raw_test_prob,
    'Calibrated_Ensemble': calibrated_test_prob,
}
# Run only if numerical summaries are needed; historical results already exist on the server.
# summaries = [evaluation_summary(name, y_test, prob) for name, prob in model_score_arrays.items()]

# ===== ADDITIONAL EXPORTS FOR REPRODUCIBILITY =====
# These per-patient outcome/prediction arrays stay in the PRIVATE output folder.
# Never commit the output folder to GitHub, even when identifiers are removed.
DEV_PRED = pd.DataFrame({
    'y_true': y_dev.to_numpy(), 'RF9': published_rf_oof_prob, 'RF25': rf25_oof_prob,
    'Weighted XGB25': xgb25_oof_prob, 'Raw ensemble': raw_dev_prob,
    'Calibrated ensemble': calibrated_dev_cf,
})
TEST_PRED = pd.DataFrame({
    'y_true': y_test.to_numpy(), 'RF9': rf2024_test_prob, 'RF25': rf25_test_prob,
    'Weighted XGB25': xgb25_test_prob, 'Raw ensemble': raw_test_prob,
    'Calibrated ensemble': calibrated_test_prob,
})
DEV_PRED.to_csv(ARGS.output_dir/'development_predictions_PRIVATE.csv.gz',index=False,compression='gzip')
TEST_PRED.to_csv(ARGS.output_dir/'holdout_predictions_PRIVATE.csv.gz',index=False,compression='gzip')
summary=[]
for group,df in [('Development OOF',DEV_PRED),('Held-out test',TEST_PRED)]:
    for model in ('RF9','RF25','Weighted XGB25','Raw ensemble','Calibrated ensemble'):
        if group=='Development OOF' and model=='Calibrated ensemble':
            # Cross-fitted Platt score is diagnostic, not fully nested model-selection validation.
            continue
        record=evaluation_summary(model,df['y_true'],df[model]);record['dataset']=group
        summary.append(record)
pd.DataFrame(summary).to_csv(ARGS.output_dir/'aggregate_metrics.csv',index=False)
meta={
    'source':'IJMS_publication_code_model.ipynb cells 2,4,6,8,10,12,14',
    'predictors_x9':FEATURES_9,'predictors_x25':FEATURES_25,
    'n_development':len(y_dev),'n_test':len(y_test),
    'RF9_final_params':rf2024_search.best_params_,
    'RF25_final_params':final_rf25_search.best_params_,
    'XGB25_parameters':XGB_PARAMS,
    'locked_fusion_weights':{'RF9':LOCKED_RF_WEIGHT,'Weighted XGB25':LOCKED_XGB_WEIGHT},
    'platt_intercept':float(final_platt.intercept_[0]),
    'platt_coefficient':float(final_platt.coef_[0,0]),
    'caveats':'Historical split reused; OOF weight selection exploratory. Raw data not published.'
}
(ARGS.output_dir/'run_manifest.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
print('Aggregate metrics (verify against manuscript):')
print(pd.DataFrame(summary).to_string(index=False))
print('Outputs written to',ARGS.output_dir,'(PRIVATE; excluded by .gitignore).')

# SAVE FITTED MODELS ONLY WHEN EXPLICITLY REQUESTED. Fitted artifacts are PRIVATE.
if ARGS.save_models:
    import joblib
    model_dir=ARGS.output_dir/'private_models'
    model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(final_rf2024,model_dir/'rf9.joblib')
    joblib.dump(final_rf25_model,model_dir/'rf25.joblib')
    joblib.dump(final_xgb25,model_dir/'weighted_xgb25.joblib')
    joblib.dump(final_platt,model_dir/'platt.joblib')
    manifest={'X9':FEATURES_9,'X25':FEATURES_25,
              'rf9_weight':LOCKED_RF_WEIGHT,'xgb25_weight':LOCKED_XGB_WEIGHT,
              'target':TARGET,'outcome_positive':'documented existing CHD',
              'warning':'Private derived model artifacts; do not post on GitHub.'}
    (model_dir/'inference_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PRIVATE models saved in',model_dir)
