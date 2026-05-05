# 📊 Retail Sales Prediction using Machine Learning

## 🚀 Overview
This project predicts retail sales using machine learning on the Walmart dataset. It helps businesses in demand forecasting, inventory planning, and decision-making.

---

## 🎯 Objective
- Predict weekly retail sales  
- Compare multiple regression models  
- Improve accuracy using feature engineering and tuning  
- Deploy model using Streamlit  

---

## 📁 Dataset
- Walmart Retail Sales Dataset (Kaggle)  
- ~6,400 records  

**Features:**
- Store  
- Date  
- Weekly Sales (Target)  
- Holiday Flag  
- Temperature  
- Fuel Price  
- CPI  
- Unemployment  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- Matplotlib  
- Power BI  
- Streamlit  

---

## 🔍 Feature Engineering
- Extracted: year, month, week  
- Lag features: lag_1, lag_2, lag_4  
- Rolling mean & standard deviation  

---

## 🤖 Models Used
- Linear Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost (Best Model)  

---

## 📊 Model Performance

| Model | RMSE | R² Score |
|------|------|---------|
| Linear Regression | ~99k | ~0.96 |
| Random Forest | ~78k | ~0.97 |
| Gradient Boosting | ~75k | ~0.98 |
| **XGBoost** | **~64k** | **~0.985** |

---

## ⚡ Hyperparameter Tuning
- Used RandomizedSearchCV  
- Reduced RMSE from ~71k → ~64k  

---

## 🖥️ Streamlit App

### Features:
- Upload dataset  
- View sales trends  
- Predict next week sales  
- Compare actual vs predicted  
- Business recommendations  

### Run App:
```bash
streamlit run app.py
