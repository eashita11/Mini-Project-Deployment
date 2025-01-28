import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import openai

# Streamlit App Configuration
st.set_page_config(page_title="Excel Data Visualization", layout="wide")

# Title
st.title("Excel Data Visualization")

# OpenAI Free Key Setup
openai.api_key = "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"

# Upload Excel File
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Load Excel Data
        df = pd.read_excel(uploaded_file)
        st.success("Excel file loaded successfully!")
        
        # Show DataFrame
        st.subheader("Preview of the Uploaded Data")
        st.dataframe(df)

        # Select Columns for Visualization
        st.subheader("Select Columns to Visualize")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

        if len(numeric_columns) == 0:
            st.error("No numeric columns available for visualization!")
        else:
            x_col = st.selectbox("Select X-axis column", numeric_columns)
            y_col = st.selectbox("Select Y-axis column", numeric_columns)

            # Plot Data
            if x_col and y_col:
                fig, ax = plt.subplots()
                ax.scatter(df[x_col], df[y_col], alpha=0.7)
                ax.set_title(f"Scatter Plot of {y_col} vs {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)

                st.pyplot(fig)

            # Generate Insights using OpenAI
            st.subheader("Generate Insights using OpenAI")
            query = st.text_input("Describe the insights you want from this data (e.g., 'Summarize trends between X and Y').")

            if query:
                prompt = f"Analyze the following data and provide insights: {query}. Data: {df[[x_col, y_col]].to_dict()}."
                try:
                    response = openai.Completion.create(
                        model="gpt-3.5-turbo",
                        prompt=prompt,
                        max_tokens=100
                    )
                    st.write(response['choices'][0]['text'].strip())
                except Exception as e:
                    st.error(f"Error with OpenAI API: {e}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
