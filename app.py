from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load model and transformers
model = joblib.load("random_forest_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

# Update CORS configuration to allow requests from your frontend domain
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "https://strokepredection.netlify.app",
            "https://your-frontend-domain.com"  # Add your frontend domain here
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Define risk thresholds
RISK_THRESHOLDS = {
    'very_low': 0.3,
    'low': 0.4,
    'medium': 0.6,
    'high': 0.8
}

def calculate_risk_factors(data):
    """Calculate individual risk factors and their contributions"""
    risk_factors = []
    risk_scores = {
        'age': 0,
        'hypertension': 0,
        'heart_disease': 0,
        'glucose': 0,
        'bmi': 0,
        'smoking': 0
    }
    
    # Age risk
    age = float(data['age'])
    if age > 60:
        risk_scores['age'] = 0.3
        risk_factors.append("Age above 60")
    elif age > 50:
        risk_scores['age'] = 0.2
        risk_factors.append("Age above 50")
    elif age > 40:
        risk_scores['age'] = 0.1
        risk_factors.append("Age above 40")

    # Hypertension risk
    if int(data['hypertension']) == 1:
        risk_scores['hypertension'] = 0.25
        risk_factors.append("Hypertension")

    # Heart disease risk
    if int(data['heart_disease']) == 1:
        risk_scores['heart_disease'] = 0.25
        risk_factors.append("Heart Disease")

    # Glucose level risk
    glucose = float(data['avg_glucose_level'])
    if glucose > 140:
        risk_scores['glucose'] = 0.2
        risk_factors.append("High Glucose Level")
    elif glucose > 120:
        risk_scores['glucose'] = 0.1
        risk_factors.append("Elevated Glucose Level")

    # BMI risk
    bmi = float(data['bmi'])
    if bmi > 30:
        risk_scores['bmi'] = 0.15
        risk_factors.append("High BMI")
    elif bmi > 25:
        risk_scores['bmi'] = 0.1
        risk_factors.append("Elevated BMI")

    # Smoking risk
    if data['smoking_status'] == 'smokes':
        risk_scores['smoking'] = 0.2
        risk_factors.append("Current Smoking")
    elif data['smoking_status'] == 'formerly smoked':
        risk_scores['smoking'] = 0.1
        risk_factors.append("Former Smoking")

    return risk_factors, risk_scores

def get_risk_category(prob, risk_factors):
    """Get detailed risk category based on probability and risk factors"""
    if prob >= RISK_THRESHOLDS['high']:
        return {
            "category": "High",
            "description": "Significant stroke risk detected",
            "recommendation": "Immediate medical consultation recommended"
        }
    elif prob >= RISK_THRESHOLDS['medium']:
        return {
            "category": "Medium",
            "description": "Moderate stroke risk detected",
            "recommendation": "Regular medical check-ups recommended"
        }
    elif prob >= RISK_THRESHOLDS['low']:
        return {
            "category": "Low",
            "description": "Slight stroke risk detected",
            "recommendation": "Monitor health indicators regularly"
        }
    else:
        return {
            "category": "Very Low",
            "description": "Minimal stroke risk detected",
            "recommendation": "Maintain healthy lifestyle"
        }

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        df = pd.DataFrame([data])

        categorical = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        numerical = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

        # Transform input data
        df_cat = pd.DataFrame(encoder.transform(df[categorical]), columns=encoder.get_feature_names_out(categorical))
        df_num = pd.DataFrame(scaler.transform(df[numerical]), columns=numerical)
        final_df = pd.concat([df_num, df_cat], axis=1)

        # Get model prediction
        probability = model.predict_proba(final_df)[0][1]  # class 1 = stroke
        prediction = int(probability >= RISK_THRESHOLDS['very_low'])

        # Calculate risk factors and their contributions
        risk_factors, risk_scores = calculate_risk_factors(data)
        
        # Get detailed risk category
        risk_category = get_risk_category(probability, risk_factors)

        # Calculate confidence intervals
        confidence_interval = 0.95  # 95% confidence interval
        std_error = np.sqrt(probability * (1 - probability) / len(final_df))
        margin_of_error = 1.96 * std_error  # 1.96 for 95% confidence

        return jsonify({
            "result": prediction,
            "confidence": round(probability * 100, 2),  # in percentage
            "confidence_interval": {
                "lower": round((probability - margin_of_error) * 100, 2),
                "upper": round((probability + margin_of_error) * 100, 2)
            },
            "risk_category": risk_category,
            "risk_factors": risk_factors,
            "risk_scores": risk_scores,
            "model_confidence": round((1 - std_error) * 100, 2)  # Model's confidence in its prediction
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

        # Calculate additional metrics
        sensitivity = matrix[3] / (matrix[3] + matrix[2])  # TP / (TP + FN)
        specificity = matrix[0] / (matrix[0] + matrix[1])  # TN / (TN + FP)
        fpr = matrix[1] / (matrix[1] + matrix[0])  # False Positive Rate
        fnr = matrix[2] / (matrix[2] + matrix[3])  # False Negative Rate
        ppv = matrix[3] / (matrix[3] + matrix[1])  # Positive Predictive Value
        npv = matrix[0] / (matrix[0] + matrix[2])  # Negative Predictive Value

        # Calculate ROC curve
        from sklearn.metrics import roc_curve
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)

        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X_final.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return jsonify({
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "auc": roc_auc_score(y_test, y_proba),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "positive_predictive_value": ppv,
            "negative_predictive_value": npv,
            "confusion_matrix": {
                "tn": int(matrix[0]),
                "fp": int(matrix[1]),
                "fn": int(matrix[2]),
                "tp": int(matrix[3]),
            },
            "roc_curve": {
                "fpr": fpr_curve.tolist(),
                "tpr": tpr_curve.tolist()
            },
            "feature_importance": feature_importance.to_dict('records')
        })

    except Exception as e:
        print("Error in metrics endpoint:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Update host to allow external connections
    app.run(host='0.0.0.0', port=5000, debug=True)
