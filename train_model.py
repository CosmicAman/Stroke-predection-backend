import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("stroke-data.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Drop 'id' column if it exists
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# Define features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Categorical and numerical columns
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# OneHotEncode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
X_encoded.index = X.index

# Scale numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols, index=X.index)

# Combine features
X_final = pd.concat([X_scaled, X_encoded], axis=1)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_final, y)

print("Original class distribution:", Counter(y))
print("Balanced class distribution:", Counter(y_balanced))

# Train-test split (after SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Train model with class weights
model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy & F1 Score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Compute ROC-AUC score
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.4f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Feature Importance
importances = model.feature_importances_
features = X_final.columns

plt.figure(figsize=(12, 6))
plt.barh(features, importances)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Save model and transformers
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model, encoder, and scaler saved successfully.")
