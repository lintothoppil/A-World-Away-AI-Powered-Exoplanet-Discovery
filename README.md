🛰️ A World Away: AI-Powered Exoplanet Discovery

An intelligent system designed to detect and classify exoplanets using machine learning techniques. This project leverages astronomical datasets and advanced models to identify potential planets beyond our solar system.

🌌 Developed for the NASA Space Apps Challenge (Local Edition hosted at Amal Jyothi College of Engineering)

📊 Performance Metrics
Metric	Score	Status
Accuracy	92.3%	⭐⭐⭐⭐⭐
Precision	91.8%	⭐⭐⭐⭐⭐
Recall	90.9%	⭐⭐⭐⭐⭐
F1-Score	91.3%	⭐⭐⭐⭐⭐
🛠️ Technology Stack
Python
Flask (Web Framework)
XGBoost (ML Model)
Pandas & NumPy
Scikit-learn
Matplotlib / Seaborn
🚀 Quick Start
# Clone the repository
git clone https://github.com/syntax-in-orbit/a-world-away.git
cd a-world-away

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

👉 Open your browser and go to:

http://localhost:5000
📁 Project Structure
a-world-away/
├── 📊 data/
│   ├── kepler_data.csv
│   └── processed/
├── 🤖 models/
│   ├── xgboost_model.pkl
│   └── preprocessor.pkl
├── 🌐 app/
│   ├── app.py
│   ├── templates/
│   └── static/
├── 📈 notebooks/
│   └── analysis.ipynb
└── 📄 requirements.txt
🎯 Features
🔍 Exoplanet classification using ML models
📊 Interactive web interface for predictions
⚡ Fast inference with pre-trained XGBoost model
📁 Clean and modular project structure
📈 Data preprocessing and feature engineering pipeline
📈 Dataset Integration
Mission	Status
Kepler	✅ Integrated
K2	✅ Integrated
TESS	✅ Integrated
🏆 Classification Results
Class	Precision	Recall	F1-Score
Confirmed	93.2%	91.5%	92.3%
Candidate	90.1%	92.3%	91.2%
False Positive	89.8%	88.9%	89.3%
📦 Dependencies
Flask
XGBoost
Scikit-learn
Pandas
NumPy
Matplotlib

Install all dependencies using:

pip install -r requirements.txt
👨‍🚀 Team
Syntax in Orbit 🚀
🌌 Inspiration

Exploring the cosmos, one algorithm at a time.

This project aims to bring AI and space exploration closer by enabling intelligent detection of exoplanets using real mission datasets.
