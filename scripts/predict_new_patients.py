#!/usr/bin/env python3
"""Apply the finalized RF9–Weighted XGB25 + Platt model to a LOCAL X25 table.
Usage:
    python scripts/predict_new_patients.py \
      --model-dir outputs/full_data/private_models \
      --input /secure/new_clinical_data.csv --output outputs/private_predictions.csv
Only accepts models saved by the trusted local 01_full_data_models.py workflow.
NO diagnosis is established by this score; external clinical validation is absent.
The CSV should contain the same 25 engineered/encoded variables as the training data.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--encoding',default='utf-8-sig',help='Input CSV encoding; use GBK for historical Chinese files')
    args=parser.parse_args()
    manifest=json.loads((args.model_dir/'inference_manifest.json').read_text(encoding='utf-8'))
    x9,x25=manifest['X9'],manifest['X25']
    if len(x9)!=9 or len(x25)!=25 or not set(x9).issubset(set(x25)):
        raise ValueError('Saved manifest contains unexpected feature schema.')
    incoming=pd.read_csv(args.input,encoding=args.encoding)
    if incoming.empty:raise ValueError('Empty incoming data')
    missing=sorted(set(x25)-set(incoming.columns))
    if missing:raise ValueError(f'Missing required engineered variables: {missing}')
    X=incoming.loc[:,x25].copy()
    for c in X:
        X[c]=pd.to_numeric(X[c],errors='raise')
    arr=X.to_numpy(dtype=float)
    if not np.isfinite(arr).all():raise ValueError('Input missing or infinite. No automatic imputation.')
    # Never load untrusted joblib files. Pickle-based formats can execute code on loading.
    rf9=joblib.load(args.model_dir/'rf9.joblib')
    xgb=joblib.load(args.model_dir/'weighted_xgb25.joblib')
    platt=joblib.load(args.model_dir/'platt.joblib')
    rf_prob=rf9.predict_proba(X[x9])[:,1]
    xgb_prob=xgb.predict_proba(X[x25])[:,1]
    w_rf=float(manifest['rf9_weight']);w_xgb=float(manifest['xgb25_weight'])
    if abs(w_rf+w_xgb-1)>1e-10 or min(w_rf,w_xgb)<0:
        raise ValueError('Invalid saved fusion weights')
    raw=w_rf*rf_prob+w_xgb*xgb_prob
    calibrated=platt.predict_proba(raw.reshape(-1,1))[:,1]
    result=pd.DataFrame({'RF9_probability':rf_prob,
                         'Weighted_XGB25_probability':xgb_prob,
                         'Raw_ensemble_probability':raw,
                         'Calibrated_CHD_probability':calibrated})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print('Saved',len(result),'local predictions to',args.output)
    print('This is identification of DOCUMENTED existing CHD, not forecasting new CHD events.')
    print('Do not upload participant-level output or fitted models to a public repository.')

if __name__=='__main__': main()
