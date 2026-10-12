# Telecom Churn Prediction

Predict which customers will leave, then choose the alert threshold by the naira cost of wrong calls instead of by accuracy.

> **The customer data is simulated** (`churn.py` generates it with a fixed seed). The cost figures below are assumptions for the demonstration, not real business numbers.

## Setup
- 7,000 customers, 26.3% churn. Split 4,200 train, 1,400 validation, 1,400 test.
- Models: logistic regression (class-weighted) and gradient boosting.
- Assumed costs: ₦45,000 lost per churner, ₦6,000 per retention offer, an offer keeps 40% of the churners it reaches.
- The threshold is chosen on the validation set only, then evaluated on the test set.

## Results
- Logistic regression AUC **0.751**, gradient boosting 0.728.
- Chosen threshold **0.575**. Test cost ₦15.25M, against ₦15.60M at the default 0.5 and ₦16.56M for doing nothing (contacting everyone costs ₦18.34M).
- That is **7.9%** below doing nothing. At the chosen threshold: 233 churners caught, 135 missed, 248 needless offers.
- Strongest signal: contract type. Two-year customers are far less likely to leave.

## Limits
- The saving is modest and depends on the assumptions. If an offer keeps fewer than about 28% of the churners it reaches, calling customers costs more than it saves.
- Simulated data. Real customer data would need its own checks.

## Run
```bash
pip install scikit-learn pandas numpy
python churn.py
```
