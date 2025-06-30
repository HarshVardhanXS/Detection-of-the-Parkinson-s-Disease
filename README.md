# 🧠 Parkinson's Disease Detection App

This interactive Streamlit web application allows you to **upload a dataset**, perform **exploratory data analysis (EDA)**, and build machine learning models to **detect Parkinson's disease** — all in your browser!

---

## 📌 Features

✅ Upload your own CSV dataset (e.g., Parkinson's voice measurements)  
✅ Automatically explore data statistics, null values, and correlations  
✅ Visualize feature distributions, pair plots, clustering, and class distributions  
✅ Train & evaluate **multiple ML models** without needing TensorFlow:
- Random Forest
- MLP (Multi-Layer Perceptron)
- SVM
- k-Nearest Neighbors

✅ View:
- Confusion matrices
- Classification reports
- Comparative performance of all models

✅ Beautiful modern UI with background image & dark overlay.

---

## 🚀 Installation

1️⃣ Clone or download this repository.

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt

3️⃣ Run the Streamlit app:

bash
Copy
Edit

📂 Example Dataset
Your dataset must contain a status column as the target label:

status=1 → Parkinson’s

status=0 → Healthy

All other numeric columns are used as features.

Example CSV snippet:

cs
Copy
Edit
MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),status
119.992,157.302,74.997,1
122.400,148.650,113.819,1
116.682,131.111,111.555,1
🎨 Screenshots
<p align="center"> <img src="https://images.unsplash.com/photo-1588776814546-ec7e194e3217" alt="Background" width="400"/> </p>
⚙️ Technologies
Python 3.8–3.13

Streamlit

pandas, numpy

scikit-learn

matplotlib, seaborn, plotly

❗ Notes
This project uses scikit-learn only — no TensorFlow needed.

Works on Python 3.8–3.13 with 64-bit interpreter.

If you encounter issues running the app directly with python script.py, always launch Streamlit apps with:

bash
Copy
Edit
streamlit run parkinsons_app.py

