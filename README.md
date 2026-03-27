<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Orbitron&size=13&duration=3000&pause=1000&color=7EB8F7&center=true&vCenter=true&width=600&lines=Detecting+worlds+beyond+our+solar+system...;Powered+by+machine+learning+%26+real+NASA+data;Kepler+%E2%80%A2+K2+%E2%80%A2+TESS+mission+datasets+integrated" alt="Typing SVG" />

<br/>

<h1>
  <img src="https://em-content.zobj.net/source/microsoft-teams/337/satellite_1f6f0-fe0f.png" width="42px" align="center" />
  &nbsp;A World Away
</h1>

<h3><em>AI-Powered Exoplanet Discovery</em></h3>

<p>
  <img src="https://img.shields.io/badge/Accuracy-92.3%25-4FC3F7?style=for-the-badge&logo=star&logoColor=white" />
  <img src="https://img.shields.io/badge/F1--Score-91.3%25-81D4FA?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-XGBoost-FF6F00?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NASA_Space_Apps-Challenge-0B3D91?style=for-the-badge&logo=nasa&logoColor=white" />
</p>

<p><sub>🏫 Local Edition hosted at <strong>Amal Jyothi College of Engineering</strong></sub></p>

<br/>

```
✦  ·  ·  ✦  ·  ·  ·  ✦  ·  ·  ✦  ·  ✦  ·  ·  ·  ✦  ·  ·  ✦  ·  ·  ·  ✦  ·  ✦
        Exploring the cosmos, one algorithm at a time.
✦  ·  ·  ✦  ·  ·  ·  ✦  ·  ·  ✦  ·  ✦  ·  ·  ·  ✦  ·  ·  ✦  ·  ·  ·  ✦  ·  ✦
```

</div>

---

## 🌌 Overview

**A World Away** is an intelligent exoplanet detection and classification system built with machine learning. By leveraging real NASA mission datasets — Kepler, K2, and TESS — the system identifies potential planets beyond our solar system with remarkable precision.

This project was developed for the **NASA Space Apps Challenge**, bridging cutting-edge AI with one of humanity's oldest pursuits: searching for new worlds.

---

## 📊 Performance Metrics

<div align="center">

| Metric | Score | Rating |
|:------:|:-----:|:------:|
| 🎯 Accuracy | **92.3%** | ⭐⭐⭐⭐⭐ |
| 🔬 Precision | **91.8%** | ⭐⭐⭐⭐⭐ |
| 🔁 Recall | **90.9%** | ⭐⭐⭐⭐⭐ |
| ⚖️ F1-Score | **91.3%** | ⭐⭐⭐⭐⭐ |

</div>

---

## 🏆 Classification Results

<div align="center">

| Class | Precision | Recall | F1-Score |
|:------|:---------:|:------:|:--------:|
| 🟢 Confirmed Planet | 93.2% | 91.5% | 92.3% |
| 🟡 Candidate | 90.1% | 92.3% | 91.2% |
| 🔴 False Positive | 89.8% | 88.9% | 89.3% |

</div>

---

## 📡 Dataset Integration

<div align="center">

| Mission | Status | Description |
|:--------|:------:|:------------|
| ![Kepler](https://img.shields.io/badge/Kepler-✅_Integrated-1a9e5f?style=flat-square) | ✅ | Primary planet-hunting photometry data |
| ![K2](https://img.shields.io/badge/K2-✅_Integrated-1a9e5f?style=flat-square) | ✅ | Extended mission across the ecliptic plane |
| ![TESS](https://img.shields.io/badge/TESS-✅_Integrated-1a9e5f?style=flat-square) | ✅ | All-sky transiting exoplanet survey |

</div>

---

## 🛠️ Technology Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)

</div>

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/syntax-in-orbit/a-world-away.git
cd a-world-away

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
source venv/bin/activate        # 🐧 Linux / 🍎 Mac
venv\Scripts\activate           # 🪟 Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the application
python app.py
```

> 🌐 Open your browser and navigate to **`http://localhost:5000`**

---

## 📁 Project Structure

```
a-world-away/
│
├── 📊 data/
│   ├── kepler_data.csv          ← Raw mission data
│   └── processed/               ← Cleaned & feature-engineered data
│
├── 🤖 models/
│   ├── xgboost_model.pkl        ← Pre-trained XGBoost classifier
│   └── preprocessor.pkl         ← Fitted data preprocessor
│
├── 🌐 app/
│   ├── app.py                   ← Flask application entry point
│   ├── templates/               ← Jinja2 HTML templates
│   └── static/                  ← CSS, JS, and assets
│
├── 📈 notebooks/
│   └── analysis.ipynb           ← EDA and model development
│
└── 📄 requirements.txt          ← Python dependencies
```

---

## 🎯 Features

- 🔍 **Exoplanet Classification** — Multi-class ML model distinguishing confirmed planets, candidates, and false positives
- 📊 **Interactive Web Interface** — Clean Flask-based UI for real-time predictions
- ⚡ **Fast Inference** — Pre-trained XGBoost model for low-latency results
- 🧹 **Preprocessing Pipeline** — Modular feature engineering and data normalization
- 📈 **Multi-Mission Support** — Unified pipeline across Kepler, K2, and TESS data

---

## 📦 Dependencies

```
flask
xgboost
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install everything with:

```bash
pip install -r requirements.txt
```

---

## 👨‍🚀 Team

<div align="center">

### 🚀 Syntax in Orbit

*Built with curiosity, caffeine, and a passion for the cosmos.*

</div>

---

## 🌌 Inspiration

<div align="center">

> *"Somewhere, something incredible is waiting to be known."*

</div>

This project was born from the belief that AI and astronomy are natural allies. By applying machine learning to real NASA photometry data, **A World Away** demonstrates how intelligent systems can accelerate one of science's most exciting frontiers — the search for planets that may one day be called home.

---

<div align="center">

<sub>Made for the NASA Space Apps Challenge · Amal Jyothi College of Engineering</sub>

<br/><br/>

![Stars](https://img.shields.io/github/stars/syntax-in-orbit/a-world-away?style=social)
![Forks](https://img.shields.io/github/forks/syntax-in-orbit/a-world-away?style=social)

</div>
