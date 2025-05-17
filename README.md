# Parkinson's Disease Detection Web App

This is a Streamlit-based web application for detecting Parkinson's Disease using machine learning.

## 🔧 Features

- Upload any Parkinson’s dataset in CSV format
- Automatic display of:
  - Dataset preview
  - Descriptive statistics
  - Correlation heatmap
  - Model training (RandomForestClassifier)
  - Evaluation metrics (Accuracy, Confusion Matrix, Classification Report)

## 🚀 How to Run

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run parkinsons_app.py
    ```

3. Open your browser to:
    ```
    http://localhost:8501
    ```

## 📂 Files

- `parkinsons_app.py`: The main Streamlit application
- `requirements.txt`: Python dependencies
- `README.md`: Project instructions

## 📌 Note

Make sure your dataset has a column named `status` where:
- `1` indicates Parkinson's
- `0` indicates Healthy