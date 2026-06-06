from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

bundle = joblib.load("models/lightgbm_model.joblib")
model = bundle["model"]

Path("reports/figures").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 8))
lgb.plot_importance(model, max_num_features=20)
plt.title("LightGBM Feature Importance - Top 20")
plt.tight_layout()
plt.savefig("reports/figures/lightgbm_feature_importance.png", dpi=160)
plt.close()

print("Saved: reports/figures/lightgbm_feature_importance.png")