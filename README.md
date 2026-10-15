# Diabetes–CHD Clinical Heterogeneity and Calibrated Ensemble Learning

This repository provides the Python implementation of the machine learning approaches used in our study:

**Clinical Heterogeneity and Calibrated Ensemble Learning for Identifying Coronary Heart Disease in Diabetic Populations Using a Large-Scale Dataset from Southwest China**

The study investigated clinical characteristics associated with documented coronary heart disease (CHD) among patients with diabetes and evaluated different machine learning strategies for handling substantial class imbalance.

## 1. Overview

The study included 120,332 patients with diabetes, of whom 3,843 had documented CHD. The analyses focused on three aspects:

- **Clinical heterogeneity:** Examination of CHD-associated clinical characteristics and age- and sex-related differences.
- **Calibrated ensemble learning:** Development and evaluation of random forest (RF), weighted XGBoost, and a heterogeneous ensemble with Platt probability calibration.
- **Alternative undersampling strategies:** Comparison of two partition-based approaches using 30 approximately balanced subsets (1:1) and 10 subsets (1:3), each incorporating four base learners and two alternative meta-learning methods.

Model performance was assessed using discrimination, probability calibration, and classification metrics.

## 2. Repository Structure

The `scripts/` directory contains the following implementations:

| Script | Description |
|---|---|
| `01_full_data_models.py` | RF9, RF25, weighted XGBoost, heterogeneous ensemble, and Platt calibration |
| `partition_30_x25.py` | Two-stage ensemble using 30 partition-based training subsets |
| `partition_10_x25.py` | Two-stage ensemble using 10 partition-based training subsets |
| `clinical_univariate_audit.py` | Exploratory univariate analysis of clinical characteristics |
| `evaluate_predictions.py` | Evaluation of ROC-AUC, average precision, calibration, and classification metrics |
| `predict_new_patients.py` | Generation of predictions using previously fitted models |

The partition-based approaches use the same 25-predictor set (X25), four first-stage algorithms (LR, RF, XGBoost, and MLP), and two second-stage alternatives (Meta-LR and Transformer). They differ primarily in the number of partitions and the resulting training class ratios.

## 3. Installation

The code is implemented in Python.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The main dependencies include NumPy, pandas, scikit-learn, XGBoost, PyTorch, and Matplotlib.

## 4. Usage

The repository provides scripts for model training, evaluation, and prediction.

The principal training scripts are:

```bash
python scripts/01_full_data_models.py --help
python scripts/partition_30_x25.py --help
python scripts/partition_10_x25.py --help
```

For basic functionality testing without the original clinical dataset, the partition-based implementations support synthetic-data smoke tests:

```bash
python scripts/partition_30_x25.py --smoke-test --cpu
python scripts/partition_10_x25.py --smoke-test --cpu
```

These tests verify basic execution and do not reproduce the numerical results reported in the manuscript.

Full model training and evaluation require appropriately formatted, authorized clinical data consistent with the study's predictor definitions and preprocessing procedures.

## 5. Data Availability

The clinical data used in this study are not publicly available because of privacy protection requirements and data-access restrictions.

**No patient-level clinical datasets, individual prediction records, or fitted models derived from the study cohort are included in this repository.**

The repository is intended to support methodological transparency and facilitate implementation of the described approaches in appropriately authorized research environments. Reproduction of the manuscript's numerical results requires access to the original study data.

## 6. Reproducibility and Limitations

The repository provides implementations of the principal machine learning and evaluation procedures described in the manuscript.

Some clinical association and subgroup analyses are not fully reconstructed in the current code release. Consequently, this repository should not be interpreted as a complete end-to-end reproduction package for every figure and table.

The study evaluated existing, documented CHD status rather than predicting future incident CHD. The reported findings also require external validation before clinical application.

## 7. Citation

If you use this code, please cite the corresponding research article.

**Manuscript:** Clinical Heterogeneity and Calibrated Ensemble Learning for Identifying Coronary Heart Disease in Diabetic Populations Using a Large-Scale Dataset from Southwest China.

*Full publication details will be added following publication.*

## 8. Contact

For questions regarding the code implementation, please contact the corresponding author through the information provided in the manuscript.
