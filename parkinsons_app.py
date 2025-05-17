import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set page configuration
st.set_page_config(
    page_title="🧠 Parkinson's Disease Detector",
    layout="wide",
    page_icon="🧠"
)

# Set vibrant background image using custom CSS
def set_background():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1588776814546-ec7e194e3217");
             background-size: cover;
             background-attachment: fixed;
             background-repeat: no-repeat;
             background-position: center;
             color: #ffffff;
         }}
         .block-container {{
             background-color: rgba(0, 0, 0, 0.6);
             padding: 2rem;
             border-radius: 1rem;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

set_background()

st.markdown("""
    <h1 style='text-align: center; color: white;'>🧬 Parkinson's Disease Detection App</h1>
    <h4 style='text-align: center; color: #ffccff;'>Upload any dataset to analyze & detect Parkinson's using machine learning</h4>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload a Parkinson's Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Summary")
    st.write(df.describe())

    st.subheader("❗ Null Values")
    st.write(df.isnull().sum())

    numeric_df = df.select_dtypes(include='number')

    # Correlation Heatmap
    st.subheader("📌 Feature Correlation Heatmap")
    fig1, ax1 = plt.subplots()
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, ax=ax1)
    st.pyplot(fig1)

    # Pie chart of target distribution
    if 'status' in df.columns:
        st.subheader("🥧 Class Distribution (Target: status)")
        pie_fig = px.pie(df, names='status', title='Status Distribution', color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(pie_fig)

    # Pair plot
    st.subheader("🔗 Feature Relationships (Pair Plot - Sampling 5 Columns)")
    sample_cols = numeric_df.columns[:5]
    fig2 = px.scatter_matrix(df[sample_cols], dimensions=sample_cols, title="Pair Plot")
    st.plotly_chart(fig2)

    # Feature Distribution
    st.subheader("📈 Feature Distributions")
    selected = st.multiselect("Select features to plot", numeric_df.columns.tolist(), default=numeric_df.columns[:3])
    for col in selected:
        fig3 = px.histogram(df, x=col, nbins=50, title=f"Distribution of {col}", color_discrete_sequence=['#00cc96'])
        st.plotly_chart(fig3)

    # Cluster Plot using 2 features
    st.subheader("🔵 Clustering Visualization")
    cluster_x = st.selectbox("X-axis", numeric_df.columns)
    cluster_y = st.selectbox("Y-axis", numeric_df.columns, index=1)
    if 'status' in df.columns:
        fig4 = px.scatter(df, x=cluster_x, y=cluster_y, color=df['status'].astype(str),
                          title=f"Clustering on {cluster_x} vs {cluster_y}",
                          color_discrete_sequence=px.colors.qualitative.G10)
        st.plotly_chart(fig4)

    # Model Section
    st.subheader("⚙️ Model Training & Evaluation")

    if 'status' in df.columns:
        X = df.drop(['status'], axis=1).select_dtypes(include='number')
        y = df['status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.metric("🎯 Accuracy", f"{acc*100:.2f}%")

        st.subheader("📉 Confusion Matrix")
        cm_fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                           color_continuous_scale="Plasma")
        st.plotly_chart(cm_fig)

        st.subheader("📋 Classification Report")
        st.write(pd.DataFrame(report).transpose())

        st.subheader("📌 Feature Importances")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
        fig5 = px.bar(imp_df, x="Feature", y="Importance", title="Feature Importances",
                      color='Importance', color_continuous_scale='Viridis')
        st.plotly_chart(fig5)

else:
    st.info("Please upload a dataset to begin analysis.")