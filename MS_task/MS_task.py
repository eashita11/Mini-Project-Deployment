import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Configuration
st.set_page_config(page_title="Excel Data Analysis", layout="wide", page_icon="📊")

# Apply custom styles
st.markdown(
    """
    <style>
        .main {
            background-color: #f8f9fa;
            color: #343a40;
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2c3e50;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: #ffffff;
        }
        .css-1aumxhk {
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("📊 Excel Data Analysis and Visualization")

# Upload Excel File
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Load Excel Data
        df = pd.read_excel(uploaded_file)
        st.success("Excel file loaded successfully!")
        
        # Show DataFrame
        st.subheader("👀 Preview of the Uploaded Data")
        st.dataframe(df, height=300)

        # Basic Statistics
        st.subheader("📈 Basic Statistics")
        if not df.empty:
            st.write(df.describe(include='all'))

        # Check for NaN Values
        st.subheader("🔍 Missing Values")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.write(missing_values[missing_values > 0])
        else:
            st.write("No missing values found!")

        # Dynamic Column Selection
        st.subheader("📊 Dynamic Column Selection")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        if len(numeric_columns) > 0:
            x_col = st.selectbox("Select X-axis column", numeric_columns)
            y_col = st.selectbox("Select Y-axis column", numeric_columns)

            # Scatter Plot
            if x_col and y_col:
                fig, ax = plt.subplots()
                ax.scatter(df[x_col], df[y_col], alpha=0.7, color='#1abc9c')
                ax.set_title(f"Scatter Plot of {y_col} vs {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("🌡️ Correlation Analysis")
        if len(numeric_columns) > 1:
            corr = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.write("Not enough numeric columns for correlation analysis.")

        # Check for Outliers
        st.subheader("⚠️ Outlier Detection")
        if len(numeric_columns) > 0:
            selected_outlier_column = st.selectbox("Select a column to check for outliers", numeric_columns)
            if selected_outlier_column:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[selected_outlier_column], ax=ax, color='#3498db')
                ax.set_title(f"Boxplot for {selected_outlier_column}")
                st.pyplot(fig)

        # Key General Insights
        st.subheader("💡 Key General Insights")
        if len(numeric_columns) > 0:
            st.write(f"**Highest value in {numeric_columns[0]}:** {df[numeric_columns[0]].max()}")
            st.write(f"**Lowest value in {numeric_columns[0]}:** {df[numeric_columns[0]].min()}")
            st.write(f"**Average value in {numeric_columns[0]}:** {df[numeric_columns[0]].mean()}")

        if len(categorical_columns) > 0:
            st.write(f"**Most frequent category in {categorical_columns[0]}:** {df[categorical_columns[0]].mode()[0]}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
