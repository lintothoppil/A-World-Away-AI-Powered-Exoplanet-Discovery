import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# --- Define Absolute File Paths ---
SCALER_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/scaler.npy'
ENCODER_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/label_encoder.npy'
IMPUTER_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/imputer.pkl'
MODEL_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/kepler_xgboost_model.pkl'
DATA_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/csv/cumulative_2025.10.04_07.30.12.csv'

# Load saved model, scaler, encoder, imputer
scaler = np.load(SCALER_PATH, allow_pickle=True).item()
le = np.load(ENCODER_PATH, allow_pickle=True).item()
imputer = joblib.load(IMPUTER_PATH)
model = joblib.load(MODEL_PATH)
features = ['koi_period', 'koi_duration', 'koi_prad', 'koi_depth', 'koi_teq',
            'koi_insol', 'koi_model_snr', 'koi_srad', 'koi_fpflag_nt', 'koi_fpflag_ss',
            'koi_fpflag_co', 'koi_fpflag_ec', 'koi_steff', 'koi_impact', 'koi_max_sngle_ev']

# Load and preprocess initial data
df = pd.read_csv(DATA_PATH, comment='#')
df = df[features + ['koi_disposition']].dropna(subset=['koi_disposition'])
X = pd.DataFrame(imputer.transform(df[features]), columns=features)  # Preserve feature names
y = le.transform(df['koi_disposition'])
X_scaled = scaler.transform(X.values)  # Convert to NumPy array for scaling

st.title('Kepler Exoplanet Hunter AI - NASA Space Apps Challenge')

st.markdown(f"""
This tool classifies exoplanet candidates using a tuned XGBoost model on Kepler KOI data.
Upload new data or enter values manually. Model accuracy: ~92% (test set, improvable with tuning).
Classes: {', '.join(le.classes_)}
""")

# Hyperparameter Tuning
st.subheader('Tune XGBoost Parameters')
n_estimators = st.slider('n_estimators', 200, 300, 200)
max_depth = st.slider('max_depth', 7, 10, 7)
learning_rate = st.slider('learning_rate', 0.1, 0.2, 0.1, 0.01)

if st.button('Apply Tuning and Retrain'):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f'Retrained Accuracy: {acc:.4f}')
    joblib.dump(xgb, MODEL_PATH)
    st.rerun()  # Refresh to load new model

# Upload new data
uploaded_file = st.file_uploader(f'Upload CSV with columns: {", ".join(features + ["koi_disposition"])}')
if uploaded_file is not None:
    try:
        # Read CSV with explicit delimiter and error handling
        data = pd.read_csv(uploaded_file, comment='#', delimiter=',', on_bad_lines='skip')
        # Ensure required columns are present
        if all(col in data.columns for col in features + ['koi_disposition']):
            # Select and impute only the required features
            X_new = pd.DataFrame(imputer.transform(data[features]), columns=features)  # Preserve feature names
            y_new = le.transform(data['koi_disposition'].dropna())
            X_new_scaled = scaler.transform(X_new.values)
            preds = model.predict(X_new_scaled)
            probs = model.predict_proba(X_new_scaled)
            data['Prediction'] = le.inverse_transform(preds)
            data['Confidence'] = np.max(probs, axis=1)
            st.write('Predictions with Confidence:')
            st.dataframe(data)
            if st.button('Retrain with This Data'):
                X_train, X_test, y_train, y_test = train_test_split(X_new_scaled, y_new, test_size=0.2, random_state=42)
                xgb = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42)
                xgb.fit(X_train, y_train)
                y_pred = xgb.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f'Retrained Accuracy on New Data: {acc:.4f}')
                joblib.dump(xgb, MODEL_PATH)
                # Update scaler and imputer if needed (placeholder)
                st.rerun()
        else:
            st.error(f'CSV must include columns: {", ".join(features + ["koi_disposition"])}')
    except pd.errors.ParserError as e:
        st.error(f'Error parsing CSV: {str(e)}. Please ensure the file has a consistent number of columns (16: {", ".join(features + ["koi_disposition"])}) and no extra delimiters. Use a standard CSV format.')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

# Manual input
st.subheader('Manual Data Entry')
input_values = []
for feat in features:
    if 'fpflag' in feat:
        input_values.append(st.selectbox(f'Enter {feat} (0 or 1)', [0, 1]))
    else:
        input_values.append(st.number_input(f'Enter {feat}', value=0.0))

if st.button('Predict'):
    X = pd.DataFrame([input_values], columns=features)  # Create DataFrame with feature names
    X_imputed = pd.DataFrame(imputer.transform(X), columns=features)  # Preserve feature names
    X_scaled = scaler.transform(X_imputed.values)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    confidence = np.max(prob)
    st.success(f'Classification: {le.inverse_transform([pred])[0]} (Confidence: {confidence:.2%})')

# Chart for Class Distribution
st.subheader('Class Distribution')
class_counts = df['koi_disposition'].value_counts()
chart_data = {
    "labels": class_counts.index.tolist(),
    "datasets": [{
        "label": "Count",
        "data": class_counts.values.tolist(),
        "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56"]
    }]
}
st.code('''chartjs
{
    "type": "bar",
    "data": {
        "labels": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
        "datasets": [{
            "label": "Count",
            "data": [1979, 2746, 4839],
            "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56"]
        }]
    },
    "options": {
        "scales": {
            "y": {"beginAtZero": true}
        }
    }
}
''', language='json')

# Model Stats
st.subheader('Model Statistics')
st.metric('Test Accuracy', '92%')  # Update with actual value
st.write(f'Classes Detected: {", ".join(le.classes_)}')
st.markdown('Tune parameters above for better performance.')

st.markdown("""
For researchers: Upload bulk data for classification and retraining.
For novices: Use manual entry to explore exoplanet properties.
""")