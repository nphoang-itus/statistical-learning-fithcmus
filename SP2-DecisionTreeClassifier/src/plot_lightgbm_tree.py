from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

bundle = joblib.load("models/lightgbm_model.joblib")
model = bundle["model"]

Path("reports/figures").mkdir(parents=True, exist_ok=True)

ax = lgb.plot_tree(
    model,
    tree_index=0,
    figsize=(24, 12),
    show_info=["split_gain", "internal_value", "leaf_count"],
)

plt.title("LightGBM Tree Visualization - Tree 0")
plt.tight_layout()
plt.savefig("reports/figures/lightgbm_tree_0.png", dpi=160)
plt.close()

print("Saved: reports/figures/lightgbm_tree_0.png")