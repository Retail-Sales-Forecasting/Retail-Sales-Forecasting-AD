# src/midpoint_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix
)

from data_loader import load_data

# ---------------------------------------------------------
# LOAD DATA (already merged with stores + holidays + transactions)
# ---------------------------------------------------------
train_df, test_df = load_data()

# Sort by time for time-series splitting
train_df = train_df.sort_values("date")

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
features = ["onpromotion", "cluster", "transactions"]   # NOW includes transactions.csv

X = train_df[features]
y_reg = train_df["sales"]
y_cls = train_df["is_holiday"]

# ---------------------------------------------------------
# TIME-BASED TRAIN / VAL / TEST SPLIT (60/20/20)
# ---------------------------------------------------------
N = len(train_df)
train_end = int(N * 0.60)
val_end   = int(N * 0.80)

# Shared split for BOTH tasks
X_train = X.iloc[:train_end]
X_val   = X.iloc[train_end:val_end]
X_test  = X.iloc[val_end:]

y_reg_train = y_reg.iloc[:train_end]
y_reg_val   = y_reg.iloc[train_end:val_end]
y_reg_test  = y_reg.iloc[val_end:]

y_cls_train = y_cls.iloc[:train_end]
y_cls_val   = y_cls.iloc[train_end:val_end]
y_cls_test  = y_cls.iloc[val_end:]

# ---------------------------------------------------------
# REQUIRED PLOT 1 — Target Distribution
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=y_cls)
plt.title("Plot 1: Holiday vs Non-Holiday Count")
plt.tight_layout()
plt.savefig("plot1_target_distribution.png")
plt.close()

# ---------------------------------------------------------
# REQUIRED PLOT 2 — Correlation Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(
    train_df[["sales","onpromotion","cluster","transactions"]].corr(),
    annot=True, cmap="coolwarm"
)
plt.title("Plot 2: Correlation Heatmap")
plt.tight_layout()
plt.savefig("plot2_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# CLASSIFICATION BASELINES (Balanced)
# ---------------------------------------------------------
cls_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42, class_weight="balanced")
}

classification_rows = []
best_cls_model = None
best_cls_f1 = -1

for name, model in cls_models.items():
    model.fit(X_train, y_cls_train)

    y_val_pred  = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_cls_val, y_val_pred)
    val_f1  = f1_score(y_cls_val, y_val_pred, zero_division=0)

    test_acc = accuracy_score(y_cls_test, y_test_pred)
    test_f1  = f1_score(y_cls_test, y_test_pred, zero_division=0)

    classification_rows.append([name, val_acc, val_f1, test_acc, test_f1])

    if val_f1 > best_cls_f1:
        best_cls_f1 = val_f1
        best_cls_model = model
        best_cls_name = name
        best_cls_test_pred = y_test_pred

# Save CSV table
cls_df = pd.DataFrame(
    classification_rows,
    columns=["Model","Val_Acc","Val_F1","Test_Acc","Test_F1"]
)
cls_df.to_csv("classification_table.csv", index=False)

# ---------------------------------------------------------
# REQUIRED PLOT 3 — Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_cls_test, best_cls_test_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Plot 3: Confusion Matrix ({best_cls_name})")
plt.tight_layout()
plt.savefig("plot3_confusion_matrix.png")
plt.close()

# ---------------------------------------------------------
# REGRESSION BASELINES — Polynomial Search (1–5)
# ---------------------------------------------------------
best_deg = None
best_rmse = float("inf")
best_poly = None
best_regressor = None

for deg in range(1, 6):
    poly = PolynomialFeatures(deg)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly   = poly.transform(X_val)

    lr = LinearRegression()
    lr.fit(X_train_poly, y_reg_train)
    y_val_pred = lr.predict(X_val_poly)
    rmse = np.sqrt(mean_squared_error(y_reg_val, y_val_pred))

    print(f"Degree {deg} RMSE = {rmse:.2f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_deg = deg
        best_poly = poly
        best_regressor = lr

# Final evaluation on regression test split
X_test_poly = best_poly.transform(X_test)
y_test_pred_reg = best_regressor.predict(X_test_poly)

mae = mean_absolute_error(y_reg_test, y_test_pred_reg)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_test_pred_reg))

# Save regression table
reg_df = pd.DataFrame(
    [[f"PolynomialDegree{best_deg}", mae, rmse]],
    columns=["Model","Val_MAE","Val_RMSE"]
)
reg_df.to_csv("regression_table.csv", index=False)

# ---------------------------------------------------------
# REQUIRED PLOT 4 — Residuals vs Predicted
# ---------------------------------------------------------
residuals = y_reg_test - y_test_pred_reg

plt.figure(figsize=(8,6))
plt.scatter(y_test_pred_reg, residuals, alpha=0.4, color="green")
plt.axhline(0, color="red", linestyle="--")
plt.title(f"Plot 4: Residuals vs Predicted (Polynomial Degree {best_deg})")
plt.xlabel("Predicted sales")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("plot4_residuals_vs_predicted.png")
plt.close()

print("\nAll midpoint outputs generated successfully (with transactions.csv included).")
