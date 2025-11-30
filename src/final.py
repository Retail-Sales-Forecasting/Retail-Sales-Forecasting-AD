# src/final_nn_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)
from sklearn.inspection import permutation_importance
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline

import tensorflow as tf
from keras import layers, models, optimizers

from data_loader import load_data


# ---------------------------------------------------------
# 1. LOAD MERGED DATA (from your data_loader.py)
# ---------------------------------------------------------
train_df, _ = load_data()
train_df = train_df.sort_values("date")  # time-based

# ---------------------------------------------------------
# 2. FEATURES & TARGETS
# ---------------------------------------------------------
candidate_features = ["onpromotion", "cluster", "transactions"]
features = [f for f in candidate_features if f in train_df.columns]

print("Using features:", features)

X = train_df[features].astype("float32")
y_reg = train_df["sales"].astype("float32")
y_cls = train_df["is_holiday"].astype("int32")

# ---------------------------------------------------------
# 3. TIME-BASED SPLIT (60/20/20) – SINGLE SPLIT FOR BOTH TASKS
# ---------------------------------------------------------
N = len(train_df)
train_end = int(N * 0.60)
val_end = int(N * 0.80)

X_train = X.iloc[:train_end].values
X_val   = X.iloc[train_end:val_end].values
X_test  = X.iloc[val_end:].values

y_reg_train = y_reg.iloc[:train_end].values
y_reg_val   = y_reg.iloc[train_end:val_end].values
y_reg_test  = y_reg.iloc[val_end:].values

y_cls_train = y_cls.iloc[:train_end].values
y_cls_val   = y_cls.iloc[train_end:val_end].values
y_cls_test  = y_cls.iloc[val_end:].values

# ---------------------------------------------------------
# 4. SCALE FEATURES FOR NEURAL NETWORKS
#    (classical models use raw features)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

input_dim = X_train_s.shape[1]

# ---------------------------------------------------------
# 5. CLASSICAL CLASSIFICATION BASELINES (LogReg + Tree)
#    with class weights to handle imbalance
# ---------------------------------------------------------
cls_models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "DecisionTreeClassifier": DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    )
}

classical_cls_results = []
best_classical_cls_model = None
best_classical_cls_name = None
best_classical_cls_val_f1 = -1
best_classical_cls_test_pred = None

for name, model in cls_models.items():
    model.fit(X_train, y_cls_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_cls_val, y_val_pred)
    val_f1 = f1_score(y_cls_val, y_val_pred, zero_division=0)

    test_acc = accuracy_score(y_cls_test, y_test_pred)
    test_f1 = f1_score(y_cls_test, y_test_pred, zero_division=0)

    classical_cls_results.append(
        [name, val_acc, val_f1, test_acc, test_f1]
    )

    if val_f1 > best_classical_cls_val_f1:
        best_classical_cls_val_f1 = val_f1
        best_classical_cls_model = model
        best_classical_cls_name = name
        best_classical_cls_test_pred = y_test_pred

print("\nClassical classification baselines:")
print(pd.DataFrame(classical_cls_results,
                   columns=["Model", "Val_Acc", "Val_F1", "Test_Acc", "Test_F1"]))


# ---------------------------------------------------------
# 6. NEURAL NETWORK FOR CLASSIFICATION (TensorFlow/Keras)
# ---------------------------------------------------------
# Compute class weights for NN
cls_weights_array = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_cls_train),
    y=y_cls_train
)
cls_class_weights = {i: w for i, w in enumerate(cls_weights_array)}
print("\nClass weights for NN classification:", cls_class_weights)

tf.random.set_seed(42)

cls_model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

cls_model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

EPOCHS_CLS = 20
BATCH_SIZE = 1024

history_cls = cls_model.fit(
    X_train_s, y_cls_train,
    validation_data=(X_val_s, y_cls_val),
    epochs=EPOCHS_CLS,
    batch_size=BATCH_SIZE,
    class_weight=cls_class_weights,
    verbose=1
)

# Predictions for metrics
y_val_pred_prob_nn = cls_model.predict(X_val_s).ravel()
y_test_pred_prob_nn = cls_model.predict(X_test_s).ravel()

y_val_pred_nn = (y_val_pred_prob_nn >= 0.5).astype(int)
y_test_pred_nn = (y_test_pred_prob_nn >= 0.5).astype(int)

val_acc_nn = accuracy_score(y_cls_val, y_val_pred_nn)
val_f1_nn = f1_score(y_cls_val, y_val_pred_nn, zero_division=0)
test_acc_nn = accuracy_score(y_cls_test, y_test_pred_nn)
test_f1_nn = f1_score(y_cls_test, y_test_pred_nn, zero_division=0)

print("\nNN classification metrics:")
print("Val Acc:", val_acc_nn, "Val F1:", val_f1_nn)
print("Test Acc:", test_acc_nn, "Test F1:", test_f1_nn)

# ---------------------------------------------------------
# REQUIRED PLOT 1 — NN Classification Learning Curve
# ---------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(history_cls.history["accuracy"], label="Train Acc")
plt.plot(history_cls.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Plot 1: NN Classification Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plot1_nn_classification_learning_curve.png")
plt.close()

# ---------------------------------------------------------
# 7. CLASSICAL REGRESSION BASELINES
#    Polynomial Regression (degrees 1–6) + Decision Tree
# ---------------------------------------------------------
reg_models = []

# Polynomial degrees 1 to 6
for degree in range(1, 7):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    pipe.fit(X_train, y_reg_train)
    y_val_pred = pipe.predict(X_val)
    mae_val = mean_absolute_error(y_reg_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_reg_val, y_val_pred))
    reg_models.append({
        "name": f"PolyDegree{degree}",
        "model": pipe,
        "val_mae": mae_val,
        "val_rmse": rmse_val
    })
    print(f"Classical regression: PolyDegree{degree} → Val RMSE = {rmse_val:.2f}")

# Decision Tree Regressor
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_reg_train)
y_val_pred_tree = tree.predict(X_val)
mae_val_tree = mean_absolute_error(y_reg_val, y_val_pred_tree)
rmse_val_tree = np.sqrt(mean_squared_error(y_reg_val, y_val_pred_tree))
reg_models.append({
    "name": "DecisionTreeRegressor",
    "model": tree,
    "val_mae": mae_val_tree,
    "val_rmse": rmse_val_tree
})
print(f"Classical regression: DecisionTreeRegressor → Val RMSE = {rmse_val_tree:.2f}")

# Pick best classical regression model based on validation RMSE
best_reg_classical = min(reg_models, key=lambda m: m["val_rmse"])
best_classical_reg_name = best_reg_classical["name"]
best_classical_reg_model = best_reg_classical["model"]
val_mae_classical = best_reg_classical["val_mae"]
val_rmse_classical = best_reg_classical["val_rmse"]

# Evaluate best classical regressor on test split
y_test_pred_reg_classical = best_classical_reg_model.predict(X_test)
test_mae_classical = mean_absolute_error(y_reg_test, y_test_pred_reg_classical)
test_rmse_classical = np.sqrt(mean_squared_error(y_reg_test, y_test_pred_reg_classical))

print("\nBest classical regression model:", best_classical_reg_name)
print("Val MAE:", val_mae_classical, "Val RMSE:", val_rmse_classical)
print("Test MAE:", test_mae_classical, "Test RMSE:", test_rmse_classical)

# ---------------------------------------------------------
# 8. NEURAL NETWORK FOR REGRESSION
# ---------------------------------------------------------
tf.random.set_seed(42)

reg_model_nn = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="linear")
])

reg_model_nn.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

EPOCHS_REG = 20
BATCH_SIZE_REG = 1024

history_reg = reg_model_nn.fit(
    X_train_s, y_reg_train,
    validation_data=(X_val_s, y_reg_val),
    epochs=EPOCHS_REG,
    batch_size=BATCH_SIZE_REG,
    verbose=1
)

y_test_pred_reg_nn = reg_model_nn.predict(X_test_s).ravel()
y_val_pred_reg_nn = reg_model_nn.predict(X_val_s).ravel()

val_mae_nn = mean_absolute_error(y_reg_val, y_val_pred_reg_nn)
val_rmse_nn = np.sqrt(mean_squared_error(y_reg_val, y_val_pred_reg_nn))

test_mae_nn = mean_absolute_error(y_reg_test, y_test_pred_reg_nn)
test_rmse_nn = np.sqrt(mean_squared_error(y_reg_test, y_test_pred_reg_nn))

print("\nNN regression metrics:")
print("Val MAE:", val_mae_nn, "Val RMSE:", val_rmse_nn)
print("Test MAE:", test_mae_nn, "Test RMSE:", test_rmse_nn)

# ---------------------------------------------------------
# REQUIRED PLOT 2 — NN Regression Learning Curve (Loss)
# ---------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(history_reg.history["loss"], label="Train Loss")
plt.plot(history_reg.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Plot 2: NN Regression Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plot2_nn_regression_learning_curve.png")
plt.close()

# ---------------------------------------------------------
# 9. TABLE 1 — CLASSIFICATION: BEST CLASSICAL VS NN
# ---------------------------------------------------------
classical_dict = {row[0]: row for row in classical_cls_results}
best_row = classical_dict[best_classical_cls_name]

classification_comparison = pd.DataFrame([
    ["Classical_" + best_classical_cls_name,
     best_row[1], best_row[2], best_row[3], best_row[4]],
    ["NN_Classifier", val_acc_nn, val_f1_nn, test_acc_nn, test_f1_nn]
], columns=["Model", "Val_Acc", "Val_F1", "Test_Acc", "Test_F1"])

classification_comparison.to_csv("table1_classification_comparison.csv",
                                 index=False)
print("\nTable 1 — Classification comparison:")
print(classification_comparison)

# ---------------------------------------------------------
# 10. TABLE 2 — REGRESSION: BEST CLASSICAL VS NN
# ---------------------------------------------------------
regression_comparison = pd.DataFrame([
    ["Classical_" + best_classical_reg_name,
     val_mae_classical, val_rmse_classical,
     test_mae_classical, test_rmse_classical],
    ["NN_Regressor",
     val_mae_nn, val_rmse_nn,
     test_mae_nn, test_rmse_nn]
], columns=["Model", "Val_MAE", "Val_RMSE", "Test_MAE", "Test_RMSE"])

regression_comparison.to_csv("table2_regression_comparison.csv",
                             index=False)
print("\nTable 2 — Regression comparison:")
print(regression_comparison)

# ---------------------------------------------------------
# 11. REQUIRED PLOT 3 — CONFUSION MATRIX (BEST FINAL MODEL)
#     choose between classical & NN based on Test F1
# ---------------------------------------------------------
if test_f1_nn >= best_row[4]:
    best_conf_model_name = "NN_Classifier"
    best_conf_y_pred = y_test_pred_nn
else:
    best_conf_model_name = "Classical_" + best_classical_cls_name
    best_conf_y_pred = best_classical_cls_test_pred

cm_final = confusion_matrix(y_cls_test, best_conf_y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm_final, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Plot 3: Confusion Matrix ({best_conf_model_name})")
plt.tight_layout()
plt.savefig("plot3_best_confusion_matrix.png")
plt.close()

# ---------------------------------------------------------
# 12. REQUIRED PLOT 4 — RESIDUALS VS PREDICTED (BEST REGRESSION)
#     choose between classical & NN based on Test RMSE
# ---------------------------------------------------------
if test_rmse_nn <= test_rmse_classical:
    best_regression_name = "NN_Regressor"
    best_y_pred_reg = y_test_pred_reg_nn
else:
    best_regression_name = "Classical_" + best_classical_reg_name
    best_y_pred_reg = y_test_pred_reg_classical

residuals = y_reg_test - best_y_pred_reg

plt.figure(figsize=(7,5))
plt.scatter(best_y_pred_reg, residuals, alpha=0.3)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title(f"Plot 4: Residuals vs Predicted ({best_regression_name})")
plt.tight_layout()
plt.savefig("plot4_best_residuals_vs_predicted.png")
plt.close()

# ---------------------------------------------------------
# 13. REQUIRED PLOT 5 — FEATURE IMPORTANCE (Permutation on best classical reg)
# ---------------------------------------------------------
perm_result = permutation_importance(
    best_classical_reg_model,
    X_test,
    y_reg_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importances_mean = perm_result.importances_mean
sorted_idx = np.argsort(importances_mean)[::-1]

plt.figure(figsize=(7,5))
plt.bar([features[i] for i in sorted_idx],
        importances_mean[sorted_idx])
plt.xlabel("Features")
plt.ylabel("Mean Importance (permutation)")
plt.title(f"Plot 5: Permutation Feature Importance ({best_classical_reg_name})")
plt.tight_layout()
plt.savefig("plot5_feature_importance.png")
plt.close()

print("\n✅ All final NN + classical outputs generated (plots + tables).")
