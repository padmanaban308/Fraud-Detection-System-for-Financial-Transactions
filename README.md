# Fraud-Detection-System-for-Financial-Transactions
An end-to-end Machine Learning fraud detection system featuring an XGBoost pipeline (trained on 555k+ records with SMOTE resampling) deployed via a FastAPI backend and a Streamlit interactive dashboard. machine-learning

An end-to-end Machine Learning fraud detection system featuring an XGBoost pipeline (trained on 555k+ records with SMOTE resampling) deployed via a FastAPI backend and a Streamlit interactive dashboard.

Software Ecosystem Language & Core :
Platform: Python 3.11+
Data Science Pipeline: pandas, numpy, scikit-learn
Imbalance Processing: Imbalanced-Learn (SMOTE)
Classifier Engine: xgboost Deployment Interface: fastapi, pydantic, joblib, streamlit, uvicorn

📈 Final Model Performance & Metrics : 
The production pipeline trained an XGBoost Classifier on over 555,719 structural transaction rows. Due to the severe inherent class imbalance (where fraud accounts for < 1% of real-world datasets), the training set was balanced using SMOTE resampling. The model was verified against a held-out 30% stratified test split (
166,716 total unseen transactions).

Test Evaluation Summary :
Overall ROC-AUC Score: 0.9957 (Exceptional class separation capability) 
Fraud Detection Recall: 0.85 (Successfully captured 85% of all active fraud attacks)

Detailed Classification Report :
TargetClass        Precision    Recall   F1-Score   Total Support 
0(Legitimate)         1.00       1.00      1.00       166,0721
(Fraudulent)          0.66       0.85      0.75       644 
Accuracy              0          0         1.00       166,716
Macro Average         0.83       0.93      0.87       166,716
Weighted Average      1.00       1.00      1.00       166,716

Confusion Matrix Breakdown :

Actual \ Predicted [Legitimate] [Fraudulent] 
[Legitimate] 165,791 281 (False Alarms) 
[Fraudulent] 94 550 (True Caught Fraud)

