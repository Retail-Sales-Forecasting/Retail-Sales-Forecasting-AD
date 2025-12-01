# src/make_submission.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from keras import layers, models, optimizers

from data_loader import load_data


# ---------------------------------------------------------
# 1. Load merged train + test from data_loader
# ---------------------------------------------------------
train_df, test_df = load_data()

# Ensure time order (good practice for this dataset)
train_df = train_df.sort_values("date")
test_df = test_df.sort_values("date")

# ---------------------------------------------------------
# 2. Feature selection (same as final_nn_pipeline)
# ---------------------------------------------------------
candidate_features = ["onpromotion", "cluster", "transactions"]
features = [f for f in candidate_features if f in train_df.columns]

print("Using features:", features)

X_full = train_df[features].astype("float32").values
y_reg_full = train_df["sales"].astype("float32").values
y_cls_full = train_df["is_holiday"].astype("int32").values

X_test = test_df[features].astype("float32").values
test_ids = test_df["id"].values

# ---------------------------------------------------------
# 3. Scale features for the NN (classical uses raw features)
# ---------------------------------------------------------
scaler = StandardScaler()
X_full_s = scaler.fit_transform(X_full)
X_test_s = scaler.transform(X_test)

input_dim = X_full_s.shape[1]

# ---------------------------------------------------------
# 4. Best CLASSIFICATION model: Logistic Regression
#    (trained on full training data, for completeness)
# ---------------------------------------------------------
cls_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
cls_model.fit(X_full, y_cls_full)

# Optional: get test classification predictions if you ever need them
y_cls_test_pred = cls_model.predict(X_test)

print("Trained best classification model: LogisticRegression")

# ---------------------------------------------------------
# 5. Best REGRESSION model: NN Regressor (same architecture as final.py)
# ---------------------------------------------------------
tf.random.set_seed(42)

reg_model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="linear")
])

reg_model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

EPOCHS = 20
BATCH_SIZE = 1024

history = reg_model.fit(
    X_full_s,
    y_reg_full,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ---------------------------------------------------------
# 6. Predict sales on test set with NN regressor
# ---------------------------------------------------------
sales_pred = reg_model.predict(X_test_s).ravel()

# Sales cannot be negative; clip to 0
sales_pred = np.maximum(sales_pred, 0.0)

# ---------------------------------------------------------
# 7. Build submission file (id, sales)
# ---------------------------------------------------------
submission = pd.DataFrame({
    "id": test_ids,
    "sales": sales_pred
})

out_path = "output.csv"
submission.to_csv(out_path, index=False)
print(f"Saved submission file to {out_path}")
