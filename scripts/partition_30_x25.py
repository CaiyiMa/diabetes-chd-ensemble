#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30 x approximately 1:1 partition-based, two-stage ensemble (fixed X25).

Stage 1: 30 non-overlapping non-CHD partitions + ALL CHD in each subset;
         4 learners (LR, RF, XGB, MLP), creating 120 probability features.
Stage 2: Meta-LR and a Transformer of 30 tokens x 4 learner probabilities.

Development:
    - 5-fold FIRST-STAGE out-of-fold probabilities and pooled ROC-AUC/AP.
    - 5-fold META-LEVEL cross-validation on the fixed first-stage OOF matrix.
      IMPORTANT: This is exploratory meta-level CV, NOT fully nested, unbiased
      end-to-end development validation. For unbiased end-to-end development
      estimates, implement nested stacking CV, or rely on external validation.
Holdout:
    - Refit each base learner on its full-development partition, predict holdout;
      apply meta models trained only on development OOF probabilities.
    - Holdout ROC-AUC/AP/Brier/log loss for family means and both meta learners.

The original historical class-wise train/test split is reproduced exactly:
a local cleaned X25 CSV (GBK), test_size=0.10, seed=42.

Run (on the user's server):
    python scripts/partition_30_x25.py

Fast engineering check (NOT for publication):
    python scripts/partition_30_x25.py --smoke-test

The script is standalone; in Jupyter you may execute main([]) after defining it,
or launch it in a shell cell:
    !python scripts/partition_30_x25.py
It reproduces the historical split from the original CSV, without requiring
any variables to be present in the current notebook session.

Requirements: numpy, pandas, scipy, scikit-learn, xgboost, torch, joblib.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- CONFIGURATION ----------
DATA_PATH = Path(os.getenv('CHD_DATA_PATH','data/private/allmax_sort.csv'))
OUTPUT_DIR = Path('outputs/partition_30')
TARGET = 'ill_CHD'
X25 = [
    'Age', 'WHtR', 'BMI', 'ill_Stroke', 'Smoking',
    'ill_Chronic_Lung_Disease', 'Sex', 'Drinking', 'MSP', 'SC',
    'Heart_Murmur', 'FBG', 'DPD', 'SPD', 'BUN', 'PLT', 'Hb',
    'Barrel_Chest', 'HDL', 'LDL', 'SOB', 'Hepatitis_B_surface_antigen',
    'WBC', 'TC/HDL', 'TG'
]
FAMILIES = ('LR', 'RF', 'XGB', 'MLP')
SEED_SPLIT = 42
SEED_PARTITION = 2026
N_PARTITIONS = 30
OOF_FOLDS = 5
INNER_FOLDS = 3
N_JOBS_SEARCH = 8

# Same X25, four base learners, grids, seeds, folds, and Transformer settings
# as the completed 10 x 1:3 run. The ONLY intended experiment difference
# is the number of disjoint majority-class partitions (30 instead of 10),
# which produces approximately 1:1 rather than 1:3 training subsets.
# Neither pipeline performs ANOVA / SelectKBest feature selection.
BASE_GRIDS = {
    'LR': {'model__penalty': ['l1', 'l2'], 'model__C': [0.1, 1.0, 10.0]},
    'RF': {'model__criterion': ['gini', 'entropy'],
           'model__max_depth': [8, 12, 16],
           'model__min_samples_split': [8, 12, 20],
           'model__n_estimators': [600], 'model__max_leaf_nodes': [100]},
    'XGB': {'model__learning_rate': [0.01, 0.02, 0.03],
            'model__n_estimators': [200, 400, 600],
            'model__max_depth': [2, 3, 4]},
    'MLP': {'model__hidden_layer_sizes': [(32,), (64,), (128,)],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate_init': [0.0005, 0.001],
            'model__activation': ['relu']}
}

# Exactly matching the 10-partition Transformer configuration:
# shared 4->d_model projection, CLS, encoder, 1-output classifier.
TRANSFORMER_CONFIG = dict(
    d_model=32, nhead=4, num_layers=1, dim_feedforward=64,
    dropout=0.10, learning_rate=0.0005, weight_decay=0.0001,
    batch_size=1024, max_epochs=60, patience=8,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_binary(y, label):
    a = np.asarray(y, dtype=int).reshape(-1)
    if not np.isin(a, [0, 1]).all() or a.sum() == 0 or a.sum() == len(a):
        raise ValueError(f'{label}: invalid binary target or a class missing.')
    return a


def metric_dict(y, p):
    y = require_binary(y, 'metric target')
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(p) != len(y) or not np.isfinite(p).all():
        raise ValueError('Missing/misaligned/nonfinite probabilities.')
    if not np.logical_and(p >= 0, p <= 1).all():
        raise ValueError('Out-of-range probability encountered.')
    pp = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        'ROC_AUC': float(roc_auc_score(y, p)),
        'AP': float(average_precision_score(y, p)),
        'Brier': float(brier_score_loss(y, p)),
        'LogLoss': float(log_loss(y, pp)),
        'Mean_probability': float(np.mean(p)),
    }


def get_data(args):
    if args.smoke_test:
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1600, n_features=25, n_informative=12,
            n_redundant=5, weights=[0.85, 0.15], flip_y=0.01,
            random_state=42,
        )
        frame = pd.DataFrame(X, columns=X25)
        frame[TARGET] = y
    else:
        if not args.data_path.exists():
            raise FileNotFoundError(
                f'Could not find {args.data_path}; provide --data-path /secure/path/allmax_sort.csv.'
            )
        frame = pd.read_csv(args.data_path, encoding='GBK')
        missing = [c for c in X25 + [TARGET] if c not in frame.columns]
        if missing:
            raise KeyError(f'Missing required original X25 columns: {missing}')

    # SAME class-wise split and combination order as Notebook Step 2.
    non_chd = frame.loc[frame[TARGET] == 0]
    chd = frame.loc[frame[TARGET] == 1]
    non_tr, non_te = train_test_split(
        non_chd, test_size=0.10, random_state=SEED_SPLIT
    )
    chd_tr, chd_te = train_test_split(
        chd, test_size=0.10, random_state=SEED_SPLIT
    )
    tr = pd.concat([non_tr, chd_tr], axis=0)
    te = pd.concat([non_te, chd_te], axis=0)
    Xdev = tr[X25].copy().reset_index(drop=True)
    Xtest = te[X25].copy().reset_index(drop=True)
    ydev = require_binary(tr[TARGET], 'development target')
    ytest = require_binary(te[TARGET], 'holdout target')
    for name, x in [('X_dev', Xdev), ('X_test', Xtest)]:
        for col in x.columns:
            x[col] = pd.to_numeric(x[col], errors='raise')
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError(f'{name} contains missing/inf; original cohort must be preprocessed.')
    if not args.smoke_test:
        assert (len(Xdev), int(ydev.sum())) == (108298, 3458)
        assert (len(Xtest), int(ytest.sum())) == (12034, 385)
    if len(set(tr.index).intersection(set(te.index))) != 0:
        raise AssertionError('Development-test original row index overlap!')
    return Xdev, ydev, Xtest, ytest


def base_pipeline(family: str, seed: int, smoke: bool = False):
    family = family.upper()
    if family == 'LR':
        return Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(
                penalty='l1', C=1.0, solver='liblinear',
                max_iter=2000, random_state=seed
            )),
        ])
    if family == 'RF':
        return Pipeline([('model', RandomForestClassifier(
            criterion='entropy', n_estimators=20 if smoke else 600,
            max_depth=12, min_samples_split=12, max_leaf_nodes=100,
            random_state=seed, n_jobs=1,
        ))])
    if family == 'XGB':
        return Pipeline([('model', XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            learning_rate=0.02, n_estimators=20 if smoke else 400,
            max_depth=3, min_child_weight=1, gamma=0,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0, reg_lambda=1, tree_method='hist',
            random_state=seed, n_jobs=1,
        ))])
    if family == 'MLP':
        return Pipeline([
            ('scale', StandardScaler()),
            ('model', MLPClassifier(
                hidden_layer_sizes=(64,), activation='relu', solver='adam',
                alpha=0.001, learning_rate_init=0.001,
                max_iter=30 if smoke else 500, early_stopping=True,
                validation_fraction=0.10, n_iter_no_change=5 if smoke else 20,
                random_state=seed,
            )),
        ])
    raise ValueError(family)


def fit_base(Xtr, ytr, Xpred, family, seed, cv_seed, args):
    t0 = time.perf_counter()
    model = base_pipeline(family, seed, smoke=args.smoke_test)
    if args.tuning_mode == 'grid':
        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=cv_seed
        )
        search = GridSearchCV(
            model, BASE_GRIDS[family], cv=inner_cv,
            scoring='roc_auc', n_jobs=args.jobs, refit=True,
            error_score='raise', pre_dispatch='2*n_jobs',
        )
        search.fit(Xtr, ytr)
        fitted = search.best_estimator_
        params = search.best_params_
        inner_auc = float(search.best_score_)
    else:
        fitted = model.fit(Xtr, ytr)
        params = {'mode': 'fixed base defaults; no ANOVA/IFS'}
        inner_auc = np.nan
    pred = np.asarray(fitted.predict_proba(Xpred)[:, 1], dtype=np.float32)
    info = {
        'Family': family, 'N': len(ytr), 'CHD': int(np.sum(ytr)),
        'NonCHD': int(len(ytr) - np.sum(ytr)),
        'Inner_CV_AUC': inner_auc, 'Params': json.dumps(params),
        'Seconds': round(time.perf_counter() - t0, 2),
    }
    return pred, fitted, info


def partitions_of_negative(ydev, n_partitions, seed):
    """Assign negative indices once, independent of outcome info beyond 0/1 membership."""
    labels = np.full(len(ydev), -1, dtype=np.int16)
    negatives = np.where(ydev == 0)[0]
    shuffled = np.random.default_rng(seed).permutation(negatives)
    for k, neg_idx in enumerate(np.array_split(shuffled, n_partitions)):
        labels[neg_idx] = k
    assert np.all(labels[ydev == 0] >= 0)
    # Positive cases are never partitioned; they are reused inside each fold.
    return labels


def subset_indices(ydev, fit_idx, partition_ids, partition_number, seed):
    fit_idx = np.asarray(fit_idx, dtype=int)
    pos = fit_idx[ydev[fit_idx] == 1]
    neg = fit_idx[(ydev[fit_idx] == 0) &
                  (partition_ids[fit_idx] == partition_number)]
    if len(pos) < 3 or len(neg) < 3:
        raise ValueError('Class shortage in a partition training set.')
    selected = np.r_[pos, neg]
    return np.random.default_rng(seed).permutation(selected)


def oof_columns(n_subsets):
    return [f'P{k:02d}_{name}' for k in range(1, n_subsets+1)
            for name in FAMILIES]


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def run_stage1(Xdev, ydev, Xtest, ytest, args, output):
    n_subsets = 2 if args.smoke_test else N_PARTITIONS
    folds = 2 if args.smoke_test else OOF_FOLDS
    partition_ids = partitions_of_negative(ydev, n_subsets, SEED_PARTITION)
    splits = list(StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=SEED_SPLIT
    ).split(Xdev, ydev))
    np.save(output/'negative_partition_ids.npy', partition_ids)
    columns = oof_columns(n_subsets)
    meta_dev = np.full((len(ydev), len(columns)), np.nan, dtype=np.float32)
    meta_test = np.full((len(ytest), len(columns)), np.nan, dtype=np.float32)
    oof_fold = np.full(len(ydev), -1, dtype=np.int16)
    details = []
    stage_dir = output/'stage1_checkpoints'
    stage_dir.mkdir(exist_ok=True)
    all_idx = np.arange(len(ydev))

    # 5-fold OOF; each fold's base learners never train on that fold's y.
    for fold_id, (fit_idx, val_idx) in enumerate(splits, 1):
        oof_fold[val_idx] = fold_id
        for partition in range(n_subsets):
            ix = subset_indices(
                ydev, fit_idx, partition_ids, partition,
                seed=SEED_PARTITION + 1000*fold_id + partition,
            )
            yy = ydev[ix]
            print(f'[OOF] fold={fold_id}/{folds} subset={partition+1}/{n_subsets} '
                  f'nonCHD={(yy==0).sum()} CHD={(yy==1).sum()}', flush=True)
            for j, family in enumerate(FAMILIES):
                k = partition * 4 + j
                ckpt = stage_dir / f'oof_fold{fold_id}_P{partition+1:02d}_{family}.npz'
                if ckpt.exists() and not args.force:
                    with np.load(ckpt, allow_pickle=False) as cached:
                        if not np.array_equal(cached['val_idx'], val_idx):
                            raise ValueError(f'Checkpoint indices mismatch: {ckpt}')
                        pred = cached['p']
                        if len(pred) != len(val_idx):
                            raise ValueError(f'Checkpoint shape mismatch: {ckpt}')
                    elapsed = np.nan
                    info = {'Family': family, 'N': len(yy),
                            'CHD': int(yy.sum()), 'NonCHD': int((yy==0).sum()),
                            'Inner_CV_AUC': np.nan,
                            'Params': 'cached predictions', 'Seconds': np.nan}
                else:
                    pred, _, info = fit_base(
                        Xdev.iloc[ix], yy, Xdev.iloc[val_idx], family,
                        seed=200000+fold_id*10000+(partition+1)*100+j,
                        cv_seed=100000+fold_id*1000+(partition+1), args=args,
                    )
                    atomic_npz(ckpt, val_idx=val_idx, p=pred)
                meta_dev[val_idx, k] = pred
                info.update(Stage='OOF', Fold=fold_id, Partition=partition+1)
                details.append(info)

    assert np.all(oof_fold > 0) and np.isfinite(meta_dev).all()
    pd.DataFrame(meta_dev, columns=columns).assign(
        dev_position=np.arange(len(ydev)), y_true=ydev, outer_fold=oof_fold
    ).to_csv(output/'oof_meta_30x4_X25.csv.gz', index=False)

    # Heldout predictions: refit base learners using development only.
    final_model_dir = output/'final_base_models'
    final_model_dir.mkdir(exist_ok=True)
    for partition in range(n_subsets):
        ix = subset_indices(ydev, all_idx, partition_ids, partition,
                            seed=SEED_PARTITION+99000+partition)
        yy = ydev[ix]
        print(f'[TEST] full-dev subset={partition+1}/{n_subsets} '
              f'nonCHD={(yy==0).sum()} CHD={(yy==1).sum()}', flush=True)
        for j, family in enumerate(FAMILIES):
            k = partition*4+j
            ckpt = stage_dir / f'test_P{partition+1:02d}_{family}.npz'
            if ckpt.exists() and not args.force:
                with np.load(ckpt, allow_pickle=False) as cached:
                    pred = cached['p']
                    if len(pred) != len(ytest):
                        raise ValueError(f'Checkpoint test shape mismatch: {ckpt}')
                info = {'Family': family, 'N': len(yy),
                        'CHD': int(yy.sum()), 'NonCHD': int((yy==0).sum()),
                        'Inner_CV_AUC': np.nan, 'Params':'cached predictions',
                        'Seconds':np.nan}
            else:
                pred, fitted, info = fit_base(
                    Xdev.iloc[ix], yy, Xtest, family,
                    seed=600000+(partition+1)*100+j,
                    cv_seed=500000+(partition+1), args=args,
                )
                atomic_npz(ckpt, p=pred)
                if args.save_models:
                    joblib.dump(fitted, final_model_dir/f'P{partition+1:02d}_{family}.joblib')
            meta_test[:, k] = pred
            info.update(Stage='TEST_REFIT', Fold=0, Partition=partition+1)
            details.append(info)

    assert np.isfinite(meta_test).all()
    pd.DataFrame(meta_test, columns=columns).assign(
        test_position=np.arange(len(ytest)), y_true=ytest
    ).to_csv(output/'test_meta_30x4_X25.csv.gz', index=False)
    pd.DataFrame(details).to_csv(output/'base_training_metadata.csv', index=False)
    return meta_dev, meta_test, oof_fold, n_subsets


class MetaTransformer(nn.Module):
    def __init__(self, n_partitions, input_dim=4, d_model=32, nhead=4,
                 num_layers=1, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.n_partitions = n_partitions
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)
        nn.init.normal_(self.cls_token, mean=0., std=0.02)

    def forward(self, x):
        # x: (batch, n_partitions, 4); n_partitions=30 in full run
        h = self.input_projection(x)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h)
        return self.classifier(self.norm(h[:, 0])).squeeze(-1)


def fit_meta_lr(Xtr, ytr, Xpred, seed=42, quick=False):
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(
            solver='liblinear', max_iter=5000, random_state=seed
        )),
    ])
    if quick:
        fitted = pipe.fit(Xtr, ytr)
        return fitted, fitted.predict_proba(Xpred)[:, 1]
    grid = {
        'model__penalty': ['l1', 'l2'],
        'model__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    }
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, scoring='roc_auc', cv=cv,
                          n_jobs=1, refit=True, error_score='raise')
    search.fit(Xtr, ytr)
    return search.best_estimator_, search.predict_proba(Xpred)[:, 1]


def _transformer_prob(model, z, device, batch_size):
    model.eval()
    data = torch.as_tensor(z, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)
    result = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result)


def fit_transformer(Xtr, ytr, Xpred, n_partitions, seed, args):
    """Train on dev-only, 10% early-stop split; output honest heldout meta probs."""
    set_seed(seed)
    cfg = TRANSFORMER_CONFIG.copy()
    if args.smoke_test:
        cfg.update(max_epochs=3, patience=2, batch_size=256,
                   d_model=16, nhead=4, dim_feedforward=32)

    # Fit learner-family scaler ONLY on meta-training samples.
    # Scaling is shared across all partition tokens per family.
    a = np.asarray(Xtr, dtype=np.float32).reshape(-1, n_partitions, 4)
    b = np.asarray(Xpred, dtype=np.float32).reshape(-1, n_partitions, 4)
    mean = a.reshape(-1, 4).mean(axis=0, keepdims=True)
    std = a.reshape(-1, 4).std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    a = ((a - mean.reshape(1,1,4)) / std.reshape(1,1,4)).astype('float32')
    b = ((b - mean.reshape(1,1,4)) / std.reshape(1,1,4)).astype('float32')

    ytr = require_binary(ytr, 'transformer meta target')
    train_idx, es_idx = train_test_split(
        np.arange(len(ytr)), test_size=0.10, stratify=ytr, random_state=seed
    )
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = MetaTransformer(
        n_partitions=n_partitions,
        d_model=cfg['d_model'], nhead=cfg['nhead'],
        num_layers=cfg['num_layers'], dim_feedforward=cfg['dim_feedforward'],
        dropout=cfg['dropout'],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(
        TensorDataset(torch.tensor(a[train_idx]), torch.tensor(ytr[train_idx],dtype=torch.float32)),
        batch_size=cfg['batch_size'], shuffle=True
    )
    best_auc, best_loss, best_epoch, no_imp = -np.inf, np.inf, 0, 0
    best_state = None
    for epoch in range(cfg['max_epochs']):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        es_prob = _transformer_prob(model, a[es_idx], device, cfg['batch_size'])
        auc = float(roc_auc_score(ytr[es_idx], es_prob))
        ll = float(log_loss(ytr[es_idx], np.clip(es_prob,1e-7,1-1e-7)))
        if auc > best_auc+1e-5 or (np.isclose(auc,best_auc) and ll<best_loss):
            best_auc,best_loss,best_epoch,no_imp = auc,ll,epoch+1,0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
            if no_imp >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    pred = _transformer_prob(model, b, device, cfg['batch_size'])
    artifact = {'state_dict': {k:v.cpu() for k,v in model.state_dict().items()},
                'mean':mean, 'std':std, 'config':cfg,
                'best_epoch':best_epoch, 'early_AUC':best_auc}
    return pred, artifact


def stage2_and_results(meta_dev, meta_test, oof_fold, n_subsets,
                       ydev, ytest, args, output):
    # Stage-one pooled OOF estimates and test predictions for all 4 families.
    rows, dev_predictions, test_predictions = [], {}, {}
    for j, family in enumerate(FAMILIES):
        dev = meta_dev[:, j::4].mean(axis=1)
        tst = meta_test[:, j::4].mean(axis=1)
        dev_predictions[f'Mean {family}'] = dev
        test_predictions[f'Mean {family}'] = tst
        rows.append(dict(Strategy=f'{n_subsets} x ~1:1 mean {family}',
                         Predictor_set='X25', Base_learners=family,
                         Dev_validation='First-stage 5-fold OOF',
                         **{'Dev_'+k:v for k,v in metric_dict(ydev,dev).items()},
                         **{'Test_'+k:v for k,v in metric_dict(ytest,tst).items()}))

    # Meta-level CV. Not nested end-to-end CV: see prominent report notes.
    meta_lr_dev = np.full(len(ydev), np.nan, dtype=float)
    trans_dev = np.full(len(ydev), np.nan, dtype=float)
    unique_fold_ids = sorted(np.unique(oof_fold).tolist())
    for fold_id in unique_fold_ids:
        trmask = oof_fold != fold_id
        valmask = oof_fold == fold_id
        print(f'[META-CV] fold={fold_id}/{len(unique_fold_ids)}', flush=True)
        _, meta_lr_dev[valmask] = fit_meta_lr(
            meta_dev[trmask], ydev[trmask], meta_dev[valmask],
            seed=3000+fold_id, quick=args.smoke_test,
        )
        trans_dev[valmask], _ = fit_transformer(
            meta_dev[trmask], ydev[trmask], meta_dev[valmask],
            n_partitions=n_subsets, seed=4000+fold_id, args=args,
        )
    assert np.isfinite(meta_lr_dev).all() and np.isfinite(trans_dev).all()

    # Final stage-two learners fit only using development OOF features.
    final_lr, lr_test = fit_meta_lr(meta_dev, ydev, meta_test,
                                    seed=5000, quick=args.smoke_test)
    joblib.dump(final_lr, output/'final_MetaLR.joblib')
    trans_test, trans_artifact = fit_transformer(
        meta_dev, ydev, meta_test, n_partitions=n_subsets,
        seed=6000, args=args,
    )
    torch.save(trans_artifact, output/'final_Transformer.pt')

    for strategy, dev, tst in [
        ('Meta-LR', meta_lr_dev, lr_test),
        ('Transformer meta', trans_dev, trans_test),
    ]:
        dev_predictions[strategy] = dev
        test_predictions[strategy] = tst
        rows.append(dict(Strategy=f'{n_subsets} x ~1:1 {strategy}',
                         Predictor_set='X25',
                         Base_learners='LR/RF/XGB/MLP + '+strategy,
                         Dev_validation='Exploratory META-CV on fixed stage-one OOF; not fully nested',
                         **{'Dev_'+k:v for k,v in metric_dict(ydev,dev).items()},
                         **{'Test_'+k:v for k,v in metric_dict(ytest,tst).items()}))

    # Outputs with every patient-level model prediction (reproducibility).
    pd.DataFrame(dev_predictions).assign(
        dev_position=np.arange(len(ydev)), y_true=ydev, meta_cv_fold=oof_fold
    ).to_csv(output/'development_prediction_scores.csv',index=False)
    pd.DataFrame(test_predictions).assign(
        test_position=np.arange(len(ytest)), y_true=ytest
    ).to_csv(output/'holdout_prediction_scores.csv',index=False)

    table = pd.DataFrame(rows)
    table.to_csv(output/'partition30_X25_metrics_for_Table_S6.csv',index=False,
                 float_format='%.8f')
    print('\n================ RESULT SUMMARY ================')
    selected = ['Strategy','Dev_ROC_AUC','Dev_AP','Test_ROC_AUC','Test_AP','Dev_validation']
    print(table[selected].to_string(index=False,float_format=lambda v:f'{v:.6f}'))
    print('\nNOTICE: Meta Dev metrics are META-LEVEL CV on FIXED OOF base features; '
          'they are not unbiased end-to-end nested-CV results. '
          'Held-out test metrics are the preferred evaluation results.')
    return table


def main(argv=None):
    # Jupyter kernels inject -f <kernel.json> into sys.argv.
    # Do not pass kernel launch arguments to our CLI parser.
    # An explicit argv list still works in notebooks and scripts.
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',type=Path,default=DATA_PATH)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT_DIR)
    parser.add_argument('--tuning-mode',choices=['grid','fixed'],default='grid',
                        help='grid matches the historical base learner tuning grids (without ANOVA/IFS).')
    parser.add_argument('--jobs',type=int,default=N_JOBS_SEARCH)
    parser.add_argument('--save-models',action='store_true',
                        help='Save all final-stage 120 sklearn base model objects.')
    parser.add_argument('--force',action='store_true',help='Recompute even if checkpoints exist.')
    parser.add_argument('--cpu',action='store_true',help='Force PyTorch CPU (default uses CUDA if available).')
    parser.add_argument('--smoke-test',action='store_true',
                        help='Synthetic quick check only: 2 folds x 2 subsets; NOT publication results.')
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.tuning_mode='fixed'
    output = args.output_dir
    if args.smoke_test:
        output = args.output_dir / 'smoke_test'
    output.mkdir(parents=True,exist_ok=True)
    if not args.force:
        old_config=output/'run_config.json'
        if old_config.exists():
            conf=json.loads(old_config.read_text())
            if (conf['tuning_mode'] != args.tuning_mode
                or conf['smoke_test'] != args.smoke_test
                or conf['dataset'] != str(args.data_path)
                or conf['n_partitions'] != (2 if args.smoke_test else N_PARTITIONS)
                or conf['features'] != X25):
                raise ValueError('Existing output directory contains checkpoints from a different '
                                 'configuration. Use a fresh --output-dir to prevent mixing predictions.')
    config={'dataset':str(args.data_path), 'n_partitions':2 if args.smoke_test else N_PARTITIONS,
            'tuning_mode':args.tuning_mode,'smoke_test':args.smoke_test,
            'features':X25,'base_families':FAMILIES,'seed_split':SEED_SPLIT,
            'seed_partition':SEED_PARTITION,'transformer':TRANSFORMER_CONFIG,
            'meta_dev_caveat':'Meta-only CV on fixed first-stage OOF, not nested end-to-end.'}
    (output/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    set_seed(SEED_PARTITION)
    Xdev, ydev, Xtest, ytest = get_data(args)
    print('DEVELOPMENT',Xdev.shape,np.bincount(ydev),flush=True)
    print('HOLDOUT',Xtest.shape,np.bincount(ytest),flush=True)
    start=time.perf_counter()
    meta_dev,meta_test,oof_fold,n_subsets = run_stage1(
        Xdev,ydev,Xtest,ytest,args,output)
    table=stage2_and_results(meta_dev,meta_test,oof_fold,n_subsets,
                             ydev,ytest,args,output)
    print(f'\nDONE: {output}; elapsed_hours={(time.perf_counter()-start)/3600:.2f}')
    return table

if __name__ == "__main__":
    main()
