# Evaluation Results

## Model: 1D CNN-LSTM with RR-Interval Features
- **Architecture**: 3-layer 1D CNN → Bidirectional LSTM (2 layers) → Feature Fusion (+ RR features) → FC Classifier
- **Dataset**: MIT-BIH Arrhythmia Database
- **Evaluation Strategy**: Patient-independent (inter-patient) split — DS1 for training, DS2 for testing
- **Loss Function**: Focal Loss (γ=2.0) with soft class weights
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 20 | **Batch Size**: 128

---

## Dataset Split

| Split | Patients (Records) | # Samples |
|---|---|---|
| Train (DS1) | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230 | 51,010 |
| Test (DS2)  | 100, 103, 104, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 217, 219, 221, 222, 228, 231, 232, 233, 234 | 54,134 |

---

## Results: Baseline vs. With RR-Interval Features

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Baseline (CNN-LSTM only) | 68% |
| **+ RR-Interval Features** | **73%** |

---

### Classification Report — Baseline (CNN-LSTM only)

```
              precision    recall  f1-score   support

           N       0.89      0.75      0.82     44653
           S       0.08      0.17      0.11      1837
           V       0.50      0.89      0.64      3384
           F       0.00      0.07      0.01       388
           Q       0.03      0.00      0.00      3872

    accuracy                           0.68     54134
   macro avg       0.30      0.38      0.32     54134
weighted avg       0.77      0.68      0.72     54134
```

**Confusion Matrix:**
```
[[33455  3526  1439  6212    21]
 [ 1405   310    68    53     1]
 [  116     7  3008   251     2]
 [  118     2   133    27   108]
 [ 2332    64  1329   143     4]]
```

---

### Classification Report — Final Model (CNN-LSTM + RR-Interval Features)

```
              precision    recall  f1-score   support

           N       0.91      0.80      0.85     44653
           S       0.20      0.26      0.23      1837
           V       0.54      0.93      0.69      3384
           F       0.01      0.20      0.02       388
           Q       0.06      0.00      0.00      3872

    accuracy                           0.73     54134
   macro avg       0.35      0.44      0.36     54134
weighted avg       0.80      0.73      0.75     54134
```

**Confusion Matrix:**
```
[[35782  1711   930  6171    59]
 [ 1259   474    40    58     6]
 [   74    24  3153   130     3]
 [  161     1   139    78     9]
 [ 1953   123  1543   248     5]]
```

---

## Class-by-Class Improvement Summary

| Class | Label | Baseline F1 | Final F1 | Change |
|---|---|---|---|---|
| N | Normal Beat | 0.82 | **0.85** | +0.03 |
| S | Supraventricular Ectopic Beat | 0.11 | **0.23** | **+0.12 ✅** |
| V | Ventricular Ectopic Beat | 0.64 | **0.69** | +0.05 |
| F | Fusion Beat | 0.01 | 0.02 | +0.01 |
| Q | Unknown Beat | 0.00 | 0.00 | — |

> **Key Insight**: The largest gain was in detecting Supraventricular Ectopic Beats (Class S), whose F1-score **doubled**. This is because premature supraventricular beats are clinically defined by a shortened RR interval — exactly what the added `pre_RR` feature captures.
