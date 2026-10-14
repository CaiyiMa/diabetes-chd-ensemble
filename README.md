# Diabetes–CHD clinical heterogeneity and calibrated ensemble learning — code release candidate

> **Status: curator-reviewed source package / RELEASE CANDIDATE, not a certified end-to-end replication.**
> No private clinical records, trained patient-derived artifacts, or individual predictions are distributed.
> The original experiments are retrospective; the outcome is **existing, documented CHD**, not prospective incident CHD.

## Scope of this repository

This repository collects the **current manuscript-relevant** workflows from four research notebooks and excludes outdated experimental pipelines. Its purpose is to show how to **train, apply, and evaluate** the study models in an authorized environment. The cleaned source data are controlled by the data provider and are not public.

| Manuscript content | Code | Verification status |
|---|---|---|
| Methods 2.1: cleaned cohort checks and historical split | `scripts/01_full_data_models.py` and partition runners | Code inspected; patient-level data unavailable here |
| Methods 2.2; Results 3.2: clinical univariate comparisons | `scripts/clinical_univariate_audit.py` | **Helper only**. Not identical to every historical test. ORs and subgroup RF analyses are NOT fully reconstructed |
| Methods 2.4; Results 3.3: RF9, RF25, Weighted XGB25, weighted ensemble, Platt | `scripts/01_full_data_models.py` | Directly organized from `IJMS_publication_code_model.ipynb` cells 2/4/6/8/10/12/14; numeric reproduction requires private data |
| Methods 2.5; Results 3.4; Supplementary Note 3 | `scripts/partition_30_x25.py` and `scripts/partition_10_x25.py` | Both copied from the **FINAL** matched X25 4-base/2-meta implementation; path defaults sanitized |
| Methods 2.6; Results 3.3; threshold and PR analyses | `scripts/evaluate_predictions.py` | New read-only evaluator for privately generated predictions; author verification against Supplementary Table S7 required |
| Supplementary Figure S7 | `scripts/plot_supp_figure_s7.py` | Reproduces **plotted tabular values**, not patient-level metrics from scratch |
| Figures 2–4; feature importance and subgroup models | See `docs/REPRODUCIBILITY_GAPS.md` | Original code includes historical, partially divergent analysis. Not claimed fully reproducible here |

**Important:** the uploaded alternative script `ijms-r1-validate_sex_age_subgroups.py` evaluated each subgroup model on its **matching subgroup of the held-out cohort**. The final manuscript instead evaluates subgroup-*trained* models on the **same entire held-out cohort**. Therefore that script is intentionally **not** included as a reproduction of Figure 3; its performance is a different estimand. Likewise, the originally submitted 33-subset/132-feature TabTransformer and its balanced 770-record test evaluation were withdrawn and are **not included as valid models**.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The full X25 experiments are expensive. CUDA-enabled PyTorch is optional for the partition Transformer; specify `--cpu` to use the CPU. Python and library-version differences can affect fitted results; see `docs/REPRODUCIBILITY_GAPS.md`.

## Private data and expected schema

Obtain authorized access to the historical complete-case input CSV (`allmax_sort.csv`, GBK encoded) from the provider, *not from this repository*. The CSV must contain all 25 **named** predictors, binary `ill_CHD` (1 = documented CHD), 120,332 unique cleaned patients, 3,843 positive outcomes, and no missing retained fields. This repository does **not** reproduce raw-data cleaning from the original 300,089 records; consult the prior publication and original documented processing workflow.

Set the data location (or pass `--data-path` explicitly):

```bash
export CHD_DATA_PATH=/private/authorized/allmax_sort.csv
```

The historical split is **class-wise train_test_split(test_size=0.10, random_state=42) followed by negative-then-positive concatenation**, rather than a new global reshuffle. All validation predictions must remain aligned by patient/order. Cohort checks should fail if a different input is provided.

## Workflow, in order

1. **Main full-data methods** (RF9, RF25, Weighted XGB25, RF9/XGB25 fusion and Platt):
   ```bash
   python scripts/01_full_data_models.py --data-path "$CHD_DATA_PATH" --output-dir outputs/full_data --jobs 8
   ```
   This fits separate folds, refits final models, outputs `aggregate_metrics.csv`, parameter manifest, and **PRIVATE** OOF/test score files (gitignored). Add `--save-models` only if you need to generate local fitted artifacts for subsequent authorized inference.
2. **Operating points and summary ROC/PR plots** using only these privately generated predictions:
   ```bash
   python scripts/evaluate_predictions.py \
       --development outputs/full_data/development_predictions_PRIVATE.csv.gz \
       --test outputs/full_data/holdout_predictions_PRIVATE.csv.gz \
       --output-dir outputs/evaluation
   ```
   Inspect numerical agreement with the paper; figures here are functional checks, not pixel-identical publication graphics.
3. **Matched undersampling (computationally expensive):**
   ```bash
   python scripts/partition_30_x25.py --data-path "$CHD_DATA_PATH" --output-dir outputs/partition_30 --tuning-mode grid --jobs 8
   python scripts/partition_10_x25.py --data-path "$CHD_DATA_PATH" --output-dir outputs/partition_10 --tuning-mode grid --jobs 8
   ```
   Both use X25, LR/RF/XGB/MLP and Meta-LR/Transformer. Five-fold first-stage OOF and exploratory meta-CV on fixed first-stage OOF are **not** fully nested validation.
4. **Apply / predict on newly supplied, fully processed X25 data** (after a main model run with `--save-models`):
   ```bash
   python scripts/predict_new_patients.py \
     --model-dir outputs/full_data/private_models \
     --input /secure/new_patients_X25.csv \
     --output outputs/private_predictions.csv
   ```
   The input must already contain the same engineered and encoded predictors. These predictions are *not* externally validated clinical diagnoses. `joblib` may execute code during loading; load only the trusted models you trained.
5. **Engineering-only synthetic dry runs:**
   ```bash
   python scripts/partition_30_x25.py --smoke-test --cpu --output-dir outputs/partition_30
   python scripts/partition_10_x25.py --smoke-test --cpu --output-dir outputs/partition_10
   ```
   These are *not* the paper's experimental numbers.
6. **Supplementary Figure S7 from reported aggregates (no private data):**
   ```bash
   python scripts/plot_supp_figure_s7.py
   ```
7. **Optional clinical count / univariate audit**: `python scripts/clinical_univariate_audit.py --data-path "$CHD_DATA_PATH"` (**not an exact historical reconstruction**).

## What can and cannot be claimed

The current manuscript's principal test values are ROC-AUC **0.707551**, AP **0.094881** for the calibrated heterogeneous ensemble on 12,034 patients (385 CHD). Calibration changed Brier **0.030568 → 0.030153** and log loss **0.136483 → 0.132585**. These are **reported reference values**, not results calculated during preparation of this public package. The test cohort was used for multiple historical comparisons, so reports are exploratory; no independent external validation is claimed. See `docs/REPRODUCIBILITY_GAPS.md` before treating this repository as ready for release.

## Privacy and publication

Never upload raw health records, identifiers, cohort membership maps, row-level probabilities/outcomes, saved fitted patient-derived model objects, Jupyter notebook outputs containing participant data, or server credentials. `.gitignore` excludes private inputs, outputs, model checkpoints, notebooks with outputs, and cache files. A license can be selected by the authors after institutional review; none is assumed.
