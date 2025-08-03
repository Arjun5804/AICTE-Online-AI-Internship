# 🔥 Fire Type Classification Using MODIS Satellite Data

This project uses satellite fire data from NASA's MODIS instruments to **predict the type of fire** detected in India.  
It classifies fires into categories such as:
- 🌿 **Vegetation Fire**
- 🏜️ **Other Static Land Source**
- 🌊 **Offshore Fire**

I built a complete machine learning pipeline and deployed it as an easy-to-use **Streamlit web app** that allows users to enter key satellite readings and instantly view the predicted fire type — with no coding required.

---

## 📊 Dataset

- **Source**: [NASA FIRMS - MODIS Active Fire Data](https://firms.modaps.eosdis.nasa.gov/)
- **Region**: India
- **Timeframe**: 2021–2023

---

## 🛠️ Tools & Technologies

- **Python 3**
- **Pandas, NumPy** – Data preprocessing
- **Scikit-learn** – ML model training
- **Random Forest Classifier** – Final model
- **Joblib** – Model serialization
- **Streamlit** – Frontend web interface
- **Google Colab** – Model development and testing
- **Virtual Environment** – Local app deployment

---

## 🚀 How It Works

1. **Data Preprocessing**: Cleaned and selected key features like brightness, brightness_T31, FRP, scan, track, and confidence.
2. **Model Training**: Trained a Random Forest classifier to predict fire types.
3. **Web App**: Created an interactive interface using Streamlit where users can input satellite data and get a fire type prediction.
4. **Deployment**: Tested locally in a virtual environment; also tested online using localtunnel for public access.

---

## 🧪 Sample Inputs

You can try the following example values in the app:

| Brightness | Brightness T31 | FRP  | Scan | Track | Confidence |
|------------|----------------|------|------|--------|------------|
| 330.0      | 295.0          | 25.0 | 1.1  | 1.0    | high       |


## 🧠 What I Learned

- How to work with real-world environmental data.
- Training and validating machine learning models for multi-class classification.
- Building a user interface to interact with ML models using Streamlit.
- Using open data to support environmental protection and decision-making.

---

## 📌 Future Work

- Improve class balance using SMOTE or class weights.
- Enhance accuracy for underrepresented classes like Offshore Fire.
- Integrate map visualizations or fire location lookup by coordinates.
