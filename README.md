# Transfusion FTTransformer Patient Representation

Experimental pipeline for learning patient representations from preoperative
clinical tabular data, with the longer-term goal of similar-patient retrieval
and transfusion decision support.

## Current stage

The first baseline predicts whether intraoperative red blood cell (RBC)
transfusion is required using an FTTransformer.

### Data split

Patient-level split by registration ID:

- Train: 744,658 surgery records / 546,988 patients
- Validation: 93,108 records / 68,374 patients
- Test: 92,798 records / 68,374 patients

No patient appears across multiple splits.

### RBC prevalence

- Train: 3.0109%
- Validation: 3.0223%
- Test: 3.0281%

## FTTransformer baseline

Best validation PR-AUC:

- 0.5324

Held-out test results:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9642 |
| PR-AUC | 0.5188 |
| Precision @ 0.5 | 0.2082 |
| Recall @ 0.5 | 0.9096 |
| F1 @ 0.5 | 0.3388 |

Training used weighted binary cross entropy because only approximately 3% of
records were positive.

## Preprocessing

Inputs currently include:

- sex
- admission department
- surgery type
- anesthesia grade
- preoperative diagnosis code
- age
- height
- weight
- HGB
- HCT
- PLT
- WBC
- APTT
- PT
- FIB
- CREA
- ALB
- eGFR

Continuous variables use training-set statistics only.

For each continuous variable:

1. obvious invalid values are converted to missing values;
2. median imputation is fitted on the training split;
3. standardization is fitted on the training split;
4. a missingness indicator is added.

Rare categorical values are grouped before embedding.

## Repository structure

```text
.
 src/
   ├── prepare_transfusion_data.py
   └── train_transfusion.py
 experiments/
   └── ft_rbc_baseline_seed42/
       ├── README.txt
       ├── best_fttransformer.pt
       ├── preprocessing_metadata.json
       ├── test_metrics.json
       └── training logs
 environment.yml
 requirements-lock.txt
 ENVIRONMENT.md
Next step
The trained FTTransformer will be used as a patient encoder:
clinical features
      ↓
FTTransformer
      ↓
patient representation
      ↓
patient similarity learning
      ↓
similar-case retrieval

Planned similarity baselines include:
1. cosine similarity between learned patient representations;
2. learned bilinear similarity;
3. pairwise / Siamese patient similarity learning.
Data availability
Clinical patient data and processed patient-level datasets are intentionally
not included in this repository.
