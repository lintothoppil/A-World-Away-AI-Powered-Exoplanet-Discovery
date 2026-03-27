import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Define Absolute File Paths ---
SCALER_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/tess_scaler.npy'
ENCODER_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/tess_label_encoder.npy'
IMPUTER_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/tess_imputer.pkl'
MODEL_PATH = 'C:/Users/lenovo/PycharmProjects/nasa/tess_xgboost_model.pkl'
DATA_PATH = '../csv/TOI_2025.10.04_09.14.10.csv'

# Load saved model, scaler, encoder, imputer
scaler = np.load(SCALER_PATH, allow_pickle=True).item()
le = np.load(ENCODER_PATH, allow_pickle=True).item()
imputer = joblib.load(IMPUTER_PATH)
model = joblib.load(MODEL_PATH)
features = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'st_teff', 'st_pmralim', 'st_pmdeclim']

# Load and preprocess initial data
df = pd.read_csv(DATA_PATH, comment='#')
df = df[features + ['tfopwg_disp']].dropna(subset=['tfopwg_disp'])
df['tfopwg_disp'] = df['tfopwg_disp'].map({'PC': 'CANDIDATE', 'KP': 'CONFIRMED', 'FP': 'FALSE POSITIVE',
                                          'APC': 'CANDIDATE', 'CP': 'CONFIRMED', 'FA': 'FALSE POSITIVE'})
y = le.transform(df['tfopwg_disp'])
X = pd.DataFrame(imputer.transform(df[features]), columns=features)  # Preserve feature names
X_scaled = scaler.transform(X)

st.title('TESS Exoplanet Hunter AI - NASA Space Apps Challenge')

st.markdown(f"""
This tool classifies exoplanet candidates using a tuned XGBoost model on TESS TOI data.
Upload new data or enter values manually. Model accuracy: ~73% (test set).
Classes: {', '.join(le.classes_)}
""")

# Hyperparameter Tuning
st.subheader('Tune XGBoost Parameters')
n_estimators = st.slider('n_estimators', 200, 400, 400)
max_depth = st.slider('max_depth', 7, 12, 12)
learning_rate = st.slider('learning_rate', 0.1, 0.3, 0.1, 0.01)

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
uploaded_file = st.file_uploader(f'Upload CSV with columns: {", ".join(features + ["tfopwg_disp"])}')
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, comment='#', delimiter=',', on_bad_lines='skip')
        if all(col in data.columns for col in features + ['tfopwg_disp']):
            data['tfopwg_disp'] = data['tfopwg_disp'].map({'PC': 'CANDIDATE', 'KP': 'CONFIRMED', 'FP': 'FALSE POSITIVE',
                                                          'APC': 'CANDIDATE', 'CP': 'CONFIRMED', 'FA': 'FALSE POSITIVE'})
            data = data.dropna(subset=['tfopwg_disp'])
            X_new = pd.DataFrame(imputer.transform(data[features]), columns=features)  # Preserve feature names
            y_new = le.transform(data['tfopwg_disp'])
            X_new_scaled = scaler.transform(X_new)
            preds = model.predict(X_new_scaled)
            probs = model.predict_proba(X_new_scaled)
            data['Prediction'] = le.inverse_transform(preds)
            data['Confidence'] = np.max(probs, axis=1)
            st.write('Predictions with Confidence:')
            st.dataframe(data)
            if st.button('Retrain with This Data'):
                X_train, X_test, y_train, y_test = train_test_split(X_new_scaled, y_new, test_size=0.2, random_state=42)
                xgb = XGBClassifier(n_estimators=400, max_depth=12, learning_rate=0.1, random_state=42)
                xgb.fit(X_train, y_train)
                y_pred = xgb.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f'Retrained Accuracy on New Data: {acc:.4f}')
                joblib.dump(xgb, MODEL_PATH)
                st.rerun()
        else:
            st.error(f'CSV must include columns: {", ".join(features + ["tfopwg_disp"])}')
    except pd.errors.ParserError as e:
        st.error(f'Error parsing CSV: {str(e)}. Ensure consistent columns and no extra delimiters.')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

# Manual input
st.subheader('Manual Data Entry')
input_values = []
for feat in features:
    if 'pmralim' in feat or 'pmdeclim' in feat:
        input_values.append(st.selectbox(f'Enter {feat} (0 or 1)', [0, 1]))
    else:
        input_values.append(st.number_input(f'Enter {feat}', value=0.0))

if st.button('Predict'):
    X = pd.DataFrame([input_values], columns=features)  # Create DataFrame with feature names
    X_imputed = pd.DataFrame(imputer.transform(X), columns=features)  # Preserve feature names
    X_scaled = scaler.transform(X_imputed)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    confidence = np.max(prob)
    st.success(f'Classification: {le.inverse_transform([pred])[0]} (Confidence: {confidence:.2%})')

# Chart for Class Distribution
st.subheader('Class Distribution')
class_counts = df['tfopwg_disp'].map({'PC': 'CANDIDATE', 'KP': 'CONFIRMED', 'FP': 'FALSE POSITIVE',
                                     'APC': 'CANDIDATE', 'CP': 'CONFIRMED', 'FA': 'FALSE POSITIVE'}).value_counts()
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