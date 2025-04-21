from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load model and transformers
model = joblib.load("random_forest_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)
CORS(app)

# Define the threshold for classification
CUSTOM_THRESHOLD = 0.3

# Categorize risk based on probability (same logic as prediction threshold)
def get_risk_category(prob):
    if prob >= 0.6:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    elif prob >= CUSTOM_THRESHOLD:
        return "Low"
    else:
        return "Very Low"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        df = pd.DataFrame([data])

        categorical = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        numerical = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

        df_cat = pd.DataFrame(encoder.transform(df[categorical]), columns=encoder.get_feature_names_out(categorical))
        df_num = pd.DataFrame(scaler.transform(df[numerical]), columns=numerical)

        final_df = pd.concat([df_num, df_cat], axis=1)

        probability = model.predict_proba(final_df)[0][1]  # class 1 = stroke

        prediction = int(probability >= CUSTOM_THRESHOLD)

        return jsonify({
            "result": prediction,
            "confidence": round(probability * 100, 2),  # in percentage
            "risk_category": get_risk_category(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        df = pd.read_csv("stroke-data.csv")
        df.dropna(inplace=True)
        if "id" in df.columns:
            df.drop("id", axis=1, inplace=True)

        X = df.drop("stroke", axis=1)
        y = df["stroke"]

        categorical = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        numerical = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

        X_encoded = pd.DataFrame(encoder.transform(X[categorical]), columns=encoder.get_feature_names_out(categorical), index=X.index)
        X_scaled = pd.DataFrame(scaler.transform(X[numerical]), columns=numerical, index=X.index)
        X_final = pd.concat([X_scaled, X_encoded], axis=1)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_final, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred).ravel()  # [tn, fp, fn, tp]

        return jsonify({
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": {
                "tn": int(matrix[0]),
                "fp": int(matrix[1]),
                "fn": int(matrix[2]),
                "tp": int(matrix[3]),
            }
        })

    except Exception as e:
        print("Error in metrics endpoint:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
