import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
    st.subheader("⚙️ Machine Learning Models Training & Evaluation")

    if 'status' in df.columns:
        X = df.drop(['status'], axis=1).select_dtypes(include='number')
        y = df['status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = []

        # ---------------- Random Forest ----------------
        with st.spinner("Training Random Forest..."):
            model_rf = RandomForestClassifier(random_state=42)
            model_rf.fit(X_train_scaled, y_train)
            y_pred_rf = model_rf.predict(X_test_scaled)

            acc_rf = accuracy_score(y_test, y_pred_rf)
            prec_rf = precision_score(y_test, y_pred_rf)
            rec_rf = recall_score(y_test, y_pred_rf)
            f1_rf = f1_score(y_test, y_pred_rf)
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

            results.append({'Model': 'Random Forest', 'Accuracy': acc_rf, 'Precision': prec_rf, 'Recall': rec_rf, 'F1-score': f1_rf})

            st.metric("🎯 Random Forest Accuracy", f"{acc_rf*100:.2f}%")
            st.subheader("📉 Random Forest Confusion Matrix")
            cm_fig = px.imshow(cm_rf, text_auto=True, labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Plasma")
            st.plotly_chart(cm_fig)

            st.subheader("📋 Random Forest Classification Report")
            st.write(pd.DataFrame(report_rf).transpose())

        # ---------------- MLP ----------------
        with st.spinner("Training MLPClassifier..."):
            mlp = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', solver='adam', max_iter=500, random_state=42)
            mlp.fit(X_train_scaled, y_train)
            y_pred_mlp = mlp.predict(X_test_scaled)

            acc_mlp = accuracy_score(y_test, y_pred_mlp)
            prec_mlp = precision_score(y_test, y_pred_mlp)
            rec_mlp = recall_score(y_test, y_pred_mlp)
            f1_mlp = f1_score(y_test, y_pred_mlp)
            cm_mlp = confusion_matrix(y_test, y_pred_mlp)
            report_mlp = classification_report(y_test, y_pred_mlp, output_dict=True)

            results.append({'Model': 'MLP', 'Accuracy': acc_mlp, 'Precision': prec_mlp, 'Recall': rec_mlp, 'F1-score': f1_mlp})

            st.metric("🎯 MLP Accuracy", f"{acc_mlp*100:.2f}%")
            st.subheader("📉 MLP Confusion Matrix")
            fig_cm_mlp = px.imshow(cm_mlp, text_auto=True, labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Viridis")
            st.plotly_chart(fig_cm_mlp)

            st.subheader("📋 MLP Classification Report")
            st.write(pd.DataFrame(report_mlp).transpose())

        # ---------------- SVM ----------------
        with st.spinner("Training SVM..."):
            svm = SVC(probability=True, random_state=42)
            svm.fit(X_train_scaled, y_train)
            y_pred_svm = svm.predict(X_test_scaled)

            acc_svm = accuracy_score(y_test, y_pred_svm)
            prec_svm = precision_score(y_test, y_pred_svm)
            rec_svm = recall_score(y_test, y_pred_svm)
            f1_svm = f1_score(y_test, y_pred_svm)
            cm_svm = confusion_matrix(y_test, y_pred_svm)
            report_svm = classification_report(y_test, y_pred_svm, output_dict=True)

            results.append({'Model': 'SVM', 'Accuracy': acc_svm, 'Precision': prec_svm, 'Recall': rec_svm, 'F1-score': f1_svm})

            st.metric("🎯 SVM Accuracy", f"{acc_svm*100:.2f}%")
            st.subheader("📉 SVM Confusion Matrix")
            fig_cm_svm = px.imshow(cm_svm, text_auto=True, labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Cividis")
            st.plotly_chart(fig_cm_svm)

            st.subheader("📋 SVM Classification Report")
            st.write(pd.DataFrame(report_svm).transpose())

        # ---------------- k-NN ----------------
        with st.spinner("Training k-NN..."):
            knn = KNeighborsClassifier()
            knn.fit(X_train_scaled, y_train)
            y_pred_knn = knn.predict(X_test_scaled)

            acc_knn = accuracy_score(y_test, y_pred_knn)
            prec_knn = precision_score(y_test, y_pred_knn)
            rec_knn = recall_score(y_test, y_pred_knn)
            f1_knn = f1_score(y_test, y_pred_knn)
            cm_knn = confusion_matrix(y_test, y_pred_knn)
            report_knn = classification_report(y_test, y_pred_knn, output_dict=True)

            results.append({'Model': 'k-NN', 'Accuracy': acc_knn, 'Precision': prec_knn, 'Recall': rec_knn, 'F1-score': f1_knn})

            st.metric("🎯 k-NN Accuracy", f"{acc_knn*100:.2f}%")
            st.subheader("📉 k-NN Confusion Matrix")
            fig_cm_knn = px.imshow(cm_knn, text_auto=True, labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Magma")
            st.plotly_chart(fig_cm_knn)

            st.subheader("📋 k-NN Classification Report")
            st.write(pd.DataFrame(report_knn).transpose())

        # ---------------- Comparison Table ----------------
        comparison_df = pd.DataFrame(results)
        st.subheader("🏆 Comparative Model Performance")
        st.dataframe(comparison_df.sort_values(by="Accuracy", ascending=False).style.highlight_max(axis=0, color='lightgreen'))

else:
    st.info("Please upload a dataset to begin analysis.")
