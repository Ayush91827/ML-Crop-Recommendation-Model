# 🌾 End-to-End Crop Recommendation Engine

An interactive machine learning web application that predicts the optimal crop to cultivate based on soil nutrient levels and meteorological parameters. Built with **Scikit-Learn**, serialized with **Joblib**, and served via **Streamlit**.

🔗 **Live Application:** [View Live Streamlit App](https://ml-crop-recommendation-modelgit-1827.streamlit.app/)


# 📌 Problem Overview
Selecting the wrong crop for a specific soil and climate profile leads to reduced yield and wasted agricultural resources. This project builds a multi-class classification pipeline to recommend the optimal crop across **22 distinct crop categories** using 7 soil and weather features.


# 🏗️ System Architecture
```text
+-------------------------------------------------------------------------------------------------+
|                                    END-TO-END SYSTEM PIPELINE                                   |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [ User Input (Streamlit UI) ]                                                                 |
|   • Soil: N, P, K, pH                                                                           |
|   • Climate: Temperature, Humidity, Rainfall                                                    |
|                                │                                                                |
|                                ▼                                                                |
|   [ Serialized Pipeline: crop_pipeline.pkl ]                                                   |
|   ┌────────────────────────────────────────┐     ┌──────────────────────────────────────────┐   |
|   │ 1. StandardScaler                      │ ──► │ 2. RandomForestClassifier                │   |
|   │    Normalizes feature scales (Z-Score) │     │    100 Decision Trees (max_depth=8)      │   |
|   └────────────────────────────────────────┘     └──────────────────────────────────────────┘   |
|                                │                                                                |
|                                ▼                                                                |
|   [ Recommendation Result ]                                                                     |
|   • Displays predicted crop (e.g., Rice, Coffee, Maize) with input summary                      |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

## 📊 Performance Metrics

The model was evaluated on an **80/20 stratified test split** (440 unseen samples across 22 classes).

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **99.55%** |
| **Classes Evaluated** | **22 unique crop categories** |
| **Precision (Macro Avg)** | **1.00** |
| **Recall (Macro Avg)** | **1.00** |
| **F1-Score (Macro Avg)** | **1.00** |

### Feature Importance Summary
* **Primary Drivers:** Rainfall, Humidity, and Nitrogen ($N$) show the highest split importance across the forest.
* **Non-Linear Rules:** Captures complex environmental thresholds that linear baselines fail to separate.

---

## 📂 Repository Structure

```text
├── .streamlit                 # Streamlit background color, font configuration
├── Assets                     # Screenshot
├── Tableau                    # Tableau Dashboard in .twbx format
├── .gitignore
├── Model Training.ipynb       # Data audit, validation, and pipeline training
├── README.md                  # Project documentation
├── Testing model.ipynb        # For testing if model works or not
├── app.py                     # Streamlit frontend application
├── requirements.txt           # Project dependencies
└── trained1stModel.pkl        # Serialized Scikit-Learn pipeline artifact
```

🚀 Local Installation & Setup


Clone the repository:

```
Bash
git clone [https://github.com/Ayush91827/ML-Crop-Recommendation-Model.git](https://github.com/Ayush91827/ML-Crop-Recommendation-Model.git)
cd ML-Crop-Recommendation-Model
```

Create and activate a virtual environment:
```
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install dependencies:
```
Bash
pip install -r requirements.txt
```

Launch the web application:
```
Bash
streamlit run app.py
```

🛠️ Tech Stack
Language: Python
Data Manipulation: Pandas, NumPy
Machine Learning: Scikit-Learn (Pipeline, StandardScaler, RandomForestClassifier)
Model Serialization: Joblib
Web UI & Deployment: Streamlit Community Cloud

## 📊 Business Intelligence & Monitoring

### Pre-Modeling Exploratory Dashboard (Training Domain)
Explores soil chemistry, climate boundaries, and crop distributions across 2,200 agricultural samples:
* **NPK Nutrient Profile Matrix:** Heatmap mapping critical Nitrogen, Phosphorus, and Potassium requirements across 22 distinct crop categories.
* **Climate Quadrant:** Temperature vs. Rainfall scatter distribution with humidity-weighted bubble scaling to isolate microclimate dependencies.
* **Soil pH Boxplots:** Interquartile range (IQR) analysis identifying tolerance boundaries and outlier resilience per crop.

🔗 **[Launch Interactive EDA Dashboard](https://public.tableau.com/views/CropRecommendationEDA/ClimateQuadrant?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

### 🖥️ Interactive Dashboard Preview

<img width="1824" height="761" alt="Screenshot 2026-08-23 002049" src="https://github.com/user-attachments/assets/5593f9fd-f1f1-4c63-b809-19f5eaf71923" />(https://public.tableau.com/views/CropRecommendationEDA/ClimateQuadrant?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)


> 💡 **Click the image above** to open and interact with the full live dashboard on Tableau Public.
