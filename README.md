# HealthTwin_AI

AI-Powered Smart Patient Triage System  

---

## Overview  

Health Twin AI is a multi-modal intelligent triage assistant designed to classify patient risk levels based on manually entered symptoms or uploaded medical records.

The system predicts:

- 🟢 Low Risk  
- 🟡 Medium Risk  
- 🔴 High Risk  

It also suggests nearby available clinics and displays their approximate distance in miles to assist patients in making faster healthcare decisions.

This project demonstrates how applied machine learning, rule-based overrides, and decision fusion logic can improve early risk identification and streamline healthcare access.

---

##  Problem Statement  

Healthcare systems face:

- Increasing patient load  
- Delays in manual triage  
- Limited early risk detection  
- Overcrowded emergency departments  

Health Twin AI provides an AI-assisted triage system that integrates physiological scoring, medical record analysis, and contextual decision logic to support smarter healthcare routing.

---

## How It Works  

1. User inputs data via Streamlit interface:
   - Age  
   - Symptoms  
   - Basic vitals  

   OR  

   Uploads a medical document:
   - `.pdf`  

2. The system processes:
   - Vitals cleaning using **Pandas & NumPy**  
   - Medical record text extraction using **PyPDF2**  
   - Keyword detection (e.g., *Prior Stroke*, *Heart Attack*)  

3. Machine Learning models evaluate:
   - **XGBoost** → Physiological risk scoring  
   - **Random Forest (Scikit-learn)** → Department classification  

4. Decision Fusion Layer:
   - Combines ML predictions  
   - Applies rule-based override flags  
   - Integrates historical medical context  

5. Output Dashboard displays:
   - Risk Level (Low / Medium / High)  
   - Contributing factors  
   - Radar chart visualization  
   - Suggested nearby clinics  
   - Distance in miles  

---

##  Features  

- Multi-modal patient input (manual + PDF upload)  
- AI-based physiological risk scoring  
- Department classification model  
- Rule-based keyword override engine  
- Custom decision fusion algorithm  
- Risk badge visualization (Low/Medium/High)  
- Radar chart vitals visualization  
- Nearby clinic suggestions with distance display  
- Clean clinical Streamlit dashboard  

---

## Tech Stack  

### Frontend / Interface
- **Streamlit** – Multi-modal web interface (Live vitals form + PDF upload)  
- **HTML/CSS** – Streamlit UI customization  

### Backend / Processing
- **Python** – Core development language  
- **Pandas** – Data preprocessing & vitals cleaning  
- **NumPy** – Numerical computations  
- **PyPDF2** – Medical record text extraction  

### Machine Learning Layer
- **XGBoost** – Physiological risk scoring  
- **Random Forest (Scikit-learn)** – Department classification  
- **Scikit-learn** – Model training & evaluation  

### NLP Module
- Rule-Based Keyword Extraction  
- Custom Override Flag Engine  

### Decision & Fusion Logic
- Custom Decision Algorithm  
- ML + Historical Context Integration  

### Visualization
- Streamlit Dashboard  
- Risk Badge (Low / Medium / High)  
- Radar Charts (Matplotlib / Plotly)  

---

## ▶️ Run the Project  

```bash
pip install -r requirements.txt
streamlit run app.py
```
---

## FutureScope

1.GPS-Based Smart Hospital Recommendation
2.Real-Time IoT Integration (Wearables + Devices)
3.Multi-Language Voice Assistant