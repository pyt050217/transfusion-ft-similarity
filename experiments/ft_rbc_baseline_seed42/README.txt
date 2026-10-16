FTTransformer RBC Baseline
==========================

Task:
Predict intraoperative RBC transfusion > 0 from preoperative features.

Split:
Patient-level split by 登记号
Train / Val / Test = 80% / 10% / 10%

Train rows:
744658

Val rows:
93108

Test rows:
92798

Train positive rate:
3.0109%

Val positive rate:
3.0223%

Test positive rate:
3.0281%

Model:
FTTransformer

Trainable parameters:
199577

Seed:
42

Epochs:
8

Best epoch:
8

Best validation PR-AUC:
0.5324227588

Test ROC-AUC:
0.9642

Test PR-AUC:
0.5188

Test Precision @ threshold 0.5:
0.2082

Test Recall @ threshold 0.5:
0.9096

Test F1 @ threshold 0.5:
0.3388

Notes:
- BCEWithLogitsLoss with positive class weight ≈ 32.21
- Continuous features include missingness indicators
- Preprocessing statistics fitted on TRAIN only
- Patient-level leakage prevented
