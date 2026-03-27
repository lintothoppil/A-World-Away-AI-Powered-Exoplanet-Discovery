import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Define Absolute File Paths ---
SCALER_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/k2_scaler.npy'
ENCODER_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/k2_label_encoder.npy'
IMPUTER_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/k2_imputer.pkl'
MODEL_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/k2_xgboost_model.pkl'
DATA_PATH = '../csv/k2pandc_2025.10.04_10.10.10.csv'

# Load saved model, scaler, encoder, imputer
scaler = np.load(SCALER_PATH, allow_pickle=True).item()
le = np.load(ENCODER_PATH, allow_pickle=True).item()
imputer = joblib.load(IMPUTER_PATH)
model = joblib.load(MODEL_PATH)
features = ['pl_orbper', 'pl_trandep', 'st_teff']

# Load and preprocess initial data
df = pd.read_csv(DATA_PATH, comment='#')
df = df[features + ['disposition']].dropna(subset=['disposition'])
# Apply the same label mapping as in k2_model.py
df['disposition'] = df['disposition'].map({'CONFIRMED': 'CONFIRMED', 'CANDIDATE': 'CANDIDATE', 'FALSE POSITIVE': 'FALSE POSITIVE', 'REFUTED': 'FALSE POSITIVE'})
y = le.transform(df['disposition'])
X = pd.DataFrame(imputer.transform(df[features]), columns=features)
X_scaled = scaler.transform(X)

st.title('K2 Exoplanet Hunter AI - NASA Space Apps Challenge')

st.markdown(f"""
This tool classifies exoplanet candidates using a tuned XGBoost model on K2 data.
Upload new data or enter values manually. Model accuracy: ~73% (test set).
Classes: {', '.join(le.classes_)}
""")

# Hyperparameter Tuning
st.subheader('Tune XGBoost Parameters')
n_estimators = st.slider('n_estimators', 200, 400, 200)
max_depth = st.slider('max_depth', 7, 12, 10)
learning_rate = st.slider('learning_rate', 0.1, 0.3, 0.3, 0.01)

if st.button('Apply Tuning and Retrain'):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f'Retrained Accuracy: {acc:.4f}')
    joblib.dump(xgb, MODEL_PATH)
    st.rerun()

# Upload new data
uploaded_file = st.file_uploader(f'Upload CSV with columns: {", ".join(features + ["disposition"])}')
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, comment='#', delimiter=',', on_bad_lines='skip')
        if all(col in data.columns for col in features + ['disposition']):
            # Apply label mapping
            data['disposition'] = data['disposition'].map({'CONFIRMED': 'CONFIRMED', 'CANDIDATE': 'CANDIDATE', 'FALSE POSITIVE': 'FALSE POSITIVE', 'REFUTED': 'FALSE POSITIVE'})
            X_new = pd.DataFrame(imputer.transform(data[features]), columns=features)
            y_new = le.transform(data['disposition'].dropna())
            X_new_scaled = scaler.transform(X_new)
            preds = model.predict(X_new_scaled)
            probs = model.predict_proba(X_new_scaled)
            data['Prediction'] = le.inverse_transform(preds)
            data['Confidence'] = np.max(probs, axis=1)
            st.write('Predictions with Confidence:')
            st.dataframe(data)
            if st.button('Retrain with This Data'):
                X_train, X_test, y_train, y_test = train_test_split(X_new_scaled, y_new, test_size=0.2, random_state=42)
                xgb = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.3, random_state=42)
                xgb.fit(X_train, y_train)
                y_pred = xgb.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f'Retrained Accuracy on New Data: {acc:.4f}')
                joblib.dump(xgb, MODEL_PATH)
                st.rerun()
        else:
            st.error(f'CSV must include columns: {", ".join(features + ["disposition"])}')
    except pd.errors.ParserError as e:
        st.error(f'Error parsing CSV: {str(e)}. Ensure consistent columns and no extra delimiters.')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

# Manual input
st.subheader('Manual Data Entry')
input_values = []
for feat in features:
    input_values.append(st.number_input(f'Enter {feat}', value=0.0))

if st.button('Predict'):
    X = pd.DataFrame([input_values], columns=features)
    X_imputed = pd.DataFrame(imputer.transform(X), columns=features)
    X_scaled = scaler.transform(X_imputed)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    confidence = np.max(prob)
    st.success(f'Classification: {le.inverse_transform([pred])[0]} (Confidence: {confidence:.2%})')

# Chart for Class Distribution
st.subheader('Class Distribution')
class_counts = df['disposition'].value_counts()
if not class_counts.empty:
    st.bar_chart(class_counts)
else:
    st.warning('No valid class distribution data available.')

# Model Stats
st.subheader('Model Statistics')
st.metric('Test Accuracy', '73%')  # Updated with actual
st.write(f'Classes Detected: {", ".join(le.classes_)}')

st.markdown("""
For researchers: Upload bulk data for classification and retraining.
For novices: Use manual entry to explore exoplanet properties.
""")