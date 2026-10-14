#!/usr/bin/env python3
"""Recalculate independent-test metrics and Figure 5 from PRIVATE saved predictions.
Designed to work with outputs of 01_full_data_models.py, not arbitrary score arrays.
Plots are newly assembled from saved predictions; review aesthetics against final Figure 5.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,brier_score_loss,log_loss,confusion_matrix

def read_scores(path):
    df=pd.read_csv(path)
    needed=['y_true','RF9','RF25','Weighted XGB25','Raw ensemble','Calibrated ensemble']
    if list(df.columns)!=needed:raise ValueError(f'Wrong column order or names in {path}: {df.columns.tolist()}')
    if df.isna().any().any():raise ValueError(f'Missing predictions: {path}')
    if not set(df.y_true.unique()).issubset({0,1}):raise ValueError('Invalid outcomes')
    if not ((df[needed[1:]]>=0)&(df[needed[1:]]<=1)).all().all():raise ValueError('Scores outside 0..1')
    return df

def choose_operating_point(y,p,target):
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    valid=np.where(fpr<=1-target+1e-12)[0]
    if len(valid)==0:raise ValueError('No candidate threshold')
    # Most sensitive point satisfying the specificity target; break ties by largest threshold.
    take=sorted(valid,key=lambda k:(tpr[k],thresholds[k]),reverse=True)[0]
    return float(thresholds[take])

def scores_at_threshold(y,p,threshold):
    yp=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
    return dict(TP=int(tp),FP=int(fp),TN=int(tn),FN=int(fn),
                sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
                PPV=tp/(tp+fp) if tp+fp else float('nan'),
                NPV=tn/(tn+fn) if tn+fn else float('nan'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/figures'))
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    dev=read_scores(args.development);test=read_scores(args.test)
    if (len(dev),int(dev.y_true.sum()),len(test),int(test.y_true.sum()))!=(108298,3458,12034,385):
        raise ValueError('Predictions do not match reported development/test counts.')
    records=[]
    for dname,frame in [('development',dev),('test',test)]:
        y=frame.y_true.to_numpy(dtype=int)
        for model in frame.columns[1:]:
            scores=frame[model].to_numpy(dtype=float)
            records.append(dict(dataset=dname,model=model,roc_auc=roc_auc_score(y,scores),
                  average_precision=average_precision_score(y,scores),
                  brier=brier_score_loss(y,scores),log_loss=log_loss(y,np.clip(scores,1e-12,1-1e-12)),
                  mean_predicted_probability=scores.mean()))
    pd.DataFrame(records).to_csv(args.output_dir/'model_metrics.csv',index=False,float_format='%.9f')
    op=[]
    for model in frame.columns[1:]:
        for specificity in (0.8,0.9):
            threshold=choose_operating_point(dev.y_true.to_numpy(),dev[model].to_numpy(),specificity)
            op.append(dict(model=model,development_target_specificity=specificity,
                          threshold=threshold,**scores_at_threshold(test.y_true.to_numpy(),test[model].to_numpy(),threshold)))
    pd.DataFrame(op).to_csv(args.output_dir/'operating_points.csv',index=False,float_format='%.9f')
    selected=['RF9','RF25','Weighted XGB25','Calibrated ensemble']
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    y=test.y_true.to_numpy(dtype=int)
    for model in selected:
        score=test[model].to_numpy()
        fpr,tpr,_=roc_curve(y,score);prec,rec,_=precision_recall_curve(y,score)
        axs[0].plot(fpr,tpr,label=f'{model} AUC={roc_auc_score(y,score):.3f}')
        axs[1].plot(rec,prec,label=f'{model} AP={average_precision_score(y,score):.3f}')
    axs[0].plot([0,1],[0,1],'--',color='gray',lw=0.8)
    axs[1].axhline(y.mean(),ls='--',color='gray',label=f'Prevalence {y.mean():.1%}')
    axs[0].set(xlabel='False-positive rate',ylabel='Sensitivity',title='Held-out ROC',xlim=(0,1),ylim=(0,1))
    axs[1].set(xlabel='Recall',ylabel='Precision',title='Held-out precision–recall',xlim=(0,1))
    for ax in axs: ax.legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for fmt in ('png','pdf','svg'):fig.savefig(args.output_dir/f'test_ROC_PR.{fmt}',dpi=300)
    plt.close(fig)
    print('Saved aggregate metrics, operating points, and ROC/PR plots. Check against manuscript.')
if __name__=='__main__':main()
