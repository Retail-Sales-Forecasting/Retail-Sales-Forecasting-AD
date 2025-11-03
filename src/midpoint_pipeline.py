
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error

from data_loader import load_data


# Load Data

train_df, _ = load_data()
print("Train shape:", train_df.shape)


# Features & Targets

features = ['onpromotion', 'cluster']
X = train_df[features]
y_reg = train_df['sales']
y_cls = train_df['is_holiday']


# Train/Validation/Test Split

X_train, X_temp, y_train_reg, y_temp_reg = train_test_split(X, y_reg, test_size=0.4, random_state=42)
X_val, X_test_reg, y_val_reg, y_test_reg = train_test_split(X_temp, y_temp_reg, test_size=0.5, random_state=42)

X_train_cls, X_temp_cls, y_train_cls, y_temp_cls = train_test_split(X, y_cls, test_size=0.4, random_state=42)
X_val_cls, X_test_cls, y_val_cls, y_test_cls = train_test_split(X_temp_cls, y_temp_cls, test_size=0.5, random_state=42)


# Classification Baselines

cls_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42)
}

cls_results = []
for name, model in cls_models.items():
    model.fit(X_train_cls, y_train_cls)
    y_val_pred = model.predict(X_val_cls)
    y_test_pred = model.predict(X_test_cls)
    val_acc = accuracy_score(y_val_cls, y_val_pred)
    val_f1 = f1_score(y_val_cls, y_val_pred, zero_division=0)
    test_acc = accuracy_score(y_test_cls, y_test_pred)
    test_f1 = f1_score(y_test_cls, y_test_pred, zero_division=0)
    cls_results.append([name, val_acc, val_f1, test_acc, test_f1])

    if name == 'DecisionTreeClassifier':
        best_cls_model = model
        best_cls_y_test_pred = y_test_pred

cls_table = pd.DataFrame(cls_results, columns=['Model', 'Val_Acc', 'Val_F1', 'Test_Acc', 'Test_F1'])
cls_table.to_csv("classification_table.csv", index=False)
print(cls_table)


# Plot 3: Confusion Matrix

cm = confusion_matrix(y_test_cls, best_cls_y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Decision Tree Classifier)')
plt.tight_layout()
plt.savefig("plot3_confusion_matrix.png")
plt.show()


# Regression: Automatic Polynomial Selection

best_degree = 1
best_rmse = float('inf')
best_model = None
best_poly = None

for degree in range(1, 6):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    lr = LinearRegression()
    lr.fit(X_train_poly, y_train_reg)
    y_val_pred = lr.predict(X_val_poly)

    rmse = np.sqrt(mean_squared_error(y_val_reg, y_val_pred))
    print(f"Degree {degree} RMSE: {rmse:.2f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_degree = degree
        best_model = lr
        best_poly = poly

print(f"Selected polynomial degree: {best_degree}")


# Evaluate Best Regression on Test Split

X_test_poly = best_poly.transform(X_test_reg)
y_test_pred = best_model.predict(X_test_poly)
mae = mean_absolute_error(y_test_reg, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred))

reg_table = pd.DataFrame([[f'PolynomialDegree{best_degree}', mae, rmse]],
                         columns=['Model', 'Val_MAE', 'Val_RMSE'])
reg_table.to_csv("regression_table.csv", index=False)
print(reg_table)


# Plot 4: Residuals vs Predicted

residuals = y_test_reg - y_test_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, residuals, alpha=0.3, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title(f"Residuals vs Predicted Sales (Polynomial Degree {best_degree})")
plt.tight_layout()
plt.savefig("plot4_residuals_vs_predicted.png")
plt.show()
