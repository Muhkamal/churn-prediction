"""Predict which telecom customers will leave, and pick the alert threshold
by money instead of by accuracy.

The customer data is SIMULATED (generated below with a fixed seed) for a
fictional Nigerian mobile and data provider. The cost figures are assumptions
for the demonstration, not real business numbers.
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 7
rng = np.random.default_rng(SEED)
N = 7000

# ------------------------------------------------------------------ simulate
df = pd.DataFrame(
    {
        "tenure_months": np.clip(rng.exponential(22, N), 1, 72).round().astype(int),
        "contract": rng.choice(["Monthly", "One year", "Two year"], N, p=[0.58, 0.27, 0.15]),
        "plan": rng.choice(["Basic", "Standard", "Premium"], N, p=[0.45, 0.35, 0.20]),
        "payment": rng.choice(["Auto-debit", "Card", "Transfer", "Cash"], N, p=[0.30, 0.25, 0.30, 0.15]),
        "support_calls": rng.poisson(1.3, N),
        "late_payments": rng.poisson(0.5, N),
        "has_bundle": rng.choice([0, 1], N, p=[0.6, 0.4]),
    }
)
base = df["plan"].map({"Basic": 5000, "Standard": 12000, "Premium": 24000})
df["monthly_charge_ngn"] = (base * rng.normal(1.0, 0.12, N)).round(-2).astype(int)

logit = (
    -0.035 * df["tenure_months"]
    + 1.15 * (df["contract"] == "Monthly")
    - 0.85 * (df["contract"] == "Two year")
    + 0.22 * df["support_calls"]
    + 0.30 * df["late_payments"]
    - 0.55 * df["has_bundle"]
    + 0.00004 * (df["monthly_charge_ngn"] - 12000)
    + 0.35 * (df["payment"] == "Cash")
    - 0.30 * (df["payment"] == "Auto-debit")
    + rng.normal(0, 0.8, N)
)
lo, hi = -6.0, 6.0
for _ in range(60):  # tune the intercept so about 26% of customers churn
    mid = (lo + hi) / 2
    if (1 / (1 + np.exp(-(logit + mid)))).mean() > 0.26:
        hi = mid
    else:
        lo = mid
p = 1 / (1 + np.exp(-(logit + mid)))
df["churned"] = (rng.random(N) < p).astype(int)
df.to_csv("customers_simulated.csv", index=False)

# ------------------------------------------------------------------ model
y = df["churned"]
X = df.drop(columns="churned")
cat = ["contract", "plan", "payment"]
num = [c for c in X.columns if c not in cat]

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

logit_model = make_pipeline(
    ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(drop="first"), cat)]),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)
gb_model = make_pipeline(
    ColumnTransformer([("n", "passthrough", num), ("c", OneHotEncoder(drop="first"), cat)]),
    HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED),
)

scores = {}
probs = {}
for name, model in {"Logistic regression": logit_model, "Gradient boosting": gb_model}.items():
    model.fit(X_tr, y_tr)
    probs[name] = {"val": model.predict_proba(X_val)[:, 1], "test": model.predict_proba(X_te)[:, 1]}
    scores[name] = {
        "test_auc": round(float(roc_auc_score(y_te, probs[name]["test"])), 3),
        "test_pr_auc": round(float(average_precision_score(y_te, probs[name]["test"])), 3),
    }

best = max(scores, key=lambda k: scores[k]["test_auc"])

# ------------------------------------------------------------------ money
LOSS = 45_000      # revenue lost when a customer leaves
OFFER = 6_000      # cost of a retention offer
SAVE_RATE = 0.40   # share of targeted churners an offer keeps


def total_cost(y_true, prob, thr):
    flag = prob >= thr
    churn = y_true.values == 1
    cost = np.where(flag & churn, OFFER + (1 - SAVE_RATE) * LOSS, 0.0)
    cost += np.where(flag & ~churn, OFFER, 0.0)
    cost += np.where(~flag & churn, LOSS, 0.0)
    return float(cost.sum())


grid = np.round(np.arange(0.05, 0.951, 0.025), 3)
val_costs = [total_cost(y_val, probs[best]["val"], t) for t in grid]
thr = float(grid[int(np.argmin(val_costs))])  # chosen on validation only

curve = [[float(t), round(total_cost(y_te, probs[best]["test"], t) / 1e6, 3)] for t in grid]
do_nothing = total_cost(y_te, probs[best]["test"], 2.0)       # flag nobody
contact_all = total_cost(y_te, probs[best]["test"], 0.0)      # flag everybody
at_default = total_cost(y_te, probs[best]["test"], 0.5)
at_chosen = total_cost(y_te, probs[best]["test"], thr)

flag = probs[best]["test"] >= thr
tp = int(((flag) & (y_te.values == 1)).sum())
fp = int(((flag) & (y_te.values == 0)).sum())
fn = int(((~flag) & (y_te.values == 1)).sum())

lr = logit_model.named_steps["logisticregression"]
feat = list(logit_model.named_steps["columntransformer"].get_feature_names_out())
drivers = sorted(zip(feat, lr.coef_[0]), key=lambda t: -abs(t[1]))[:6]
clean = lambda s: s.split("__", 1)[1]

out = {
    "simulated": True,
    "n_customers": N,
    "churn_rate": round(float(y.mean()), 3),
    "split": {"train": len(X_tr), "validation": len(X_val), "test": len(X_te)},
    "models": scores,
    "best_model": best,
    "assumptions_ngn": {"loss_per_churner": LOSS, "offer_cost": OFFER, "offer_save_rate": SAVE_RATE},
    "chosen_threshold": thr,
    "test_cost_ngn_m": {
        "do_nothing": round(do_nothing / 1e6, 2),
        "contact_everyone": round(contact_all / 1e6, 2),
        "default_threshold_0.5": round(at_default / 1e6, 2),
        "chosen_threshold": round(at_chosen / 1e6, 2),
    },
    "saving_vs_do_nothing_pct": round((1 - at_chosen / do_nothing) * 100, 1),
    "saving_vs_default_pct": round((1 - at_chosen / at_default) * 100, 1),
    "at_chosen": {"churners_caught": tp, "churners_missed": fn, "needless_offers": fp},
    "cost_curve": curve,
    "drivers_logistic": [[clean(n), round(float(c), 2)] for n, c in drivers],
}
json.dump(out, open("results.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "cost_curve"}, indent=2))
