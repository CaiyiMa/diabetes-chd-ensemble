#!/usr/bin/env python3
"""Descriptive/univariate clinical audit for cleaned historical X25 (exploratory helper).
This is a newly organized helper derived from the manuscript's Methods 2.1–2.2,
not a byte-for-byte reproduction of historical Figure 2 ORs, subgroup RFs, or Figure 4.
Requires verification of variable encoding and test definitions before citation.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency,mannwhitneyu
from statsmodels.stats.multitest import multipletests
X9=['Age','WHtR','BMI','ill_Stroke','Smoking','ill_Chronic_Lung_Disease','Sex','Drinking','MSP']
X25=X9+['SC','Heart_Murmur','FBG','DPD','SPD','BUN','PLT','Hb','Barrel_Chest','HDL','LDL','SOB','Hepatitis_B_surface_antigen','WBC','TC/HDL','TG']
CATEGORICAL={'ill_Stroke','Smoking','ill_Chronic_Lung_Disease','Sex','Drinking','Heart_Murmur','Barrel_Chest','SOB','Hepatitis_B_surface_antigen'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/clinical_univariate'))
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data_path,encoding='GBK')
    assert set(X25+['ill_CHD']).issubset(df.columns)
    assert len(df)==120332 and df.ill_CHD.sum()==3843,'Expected manuscript cohort'
    # For tables S1–S3 use the DEVELOPMENT cohort exclusively.
    from sklearn.model_selection import train_test_split
    a=df[df.ill_CHD==0];b=df[df.ill_CHD==1]
    a,_=train_test_split(a,test_size=.1,random_state=42)
    b,_=train_test_split(b,test_size=.1,random_state=42)
    dev=pd.concat([a,b]); assert len(dev)==108298
    rows=[]
    for col in X25:
        pos=dev.loc[dev.ill_CHD==1,col].dropna(); neg=dev.loc[dev.ill_CHD==0,col].dropna()
        if col in CATEGORICAL:
            tab=pd.crosstab(dev[col],dev.ill_CHD)
            if tab.shape[0]<2 or tab.shape[1]<2:
                pv=np.nan
            else:
                pv=chi2_contingency(tab)[1]
            test='Pearson chi-square (verify low expected counts)'
        else:
            pv=mannwhitneyu(pos,neg,alternative='two-sided').pvalue
            test='Mann–Whitney/Wilcoxon rank-sum'
        rows.append({'variable':col,'type':'categorical' if col in CATEGORICAL else 'continuous',
                     'n_CHD':len(pos),'n_nonCHD':len(neg),'CHD_median':pos.median() if col not in CATEGORICAL else np.nan,
                     'nonCHD_median':neg.median() if col not in CATEGORICAL else np.nan,
                     'test':test,'p_raw':pv})
    result=pd.DataFrame(rows)
    good=result.p_raw.notna()
    result['p_BH_FDR']=np.nan
    result.loc[good,'p_BH_FDR']=multipletests(result.loc[good,'p_raw'],method='fdr_bh')[1]
    result.to_csv(args.output_dir/'exploratory_univariate_variables.csv',index=False)
    print('Generated DEVELOPMENT-only univariate audit; validate against original S1–S3 before publication.')
if __name__=='__main__':main()
