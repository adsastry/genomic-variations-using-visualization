# genomic-variations-using-visualization
This repository was made for an internship project.
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/aditi/Downloads/test_rows.csv")  # Replace with your dataset
    return df

df = load_data()

# Sidebar options
st.sidebar.title("Genomic Data Analysis")
selected_feature = st.sidebar.selectbox("Select Feature to Visualize", df.columns)

# Main Title
st.title("Genomic Data Visualization Dashboard")

# Display Data
st.write("### Preview of Data")
st.dataframe(df.head())

# Univariate Analysis
st.write(f"### Distribution of {selected_feature}")
fig, ax = plt.subplots()
sns.histplot(df[selected_feature], kde=True, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.write("### Feature Correlation Matrix")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Scatter Plot for Multivariate Analysis
if len(df.columns) > 2:
    feature_x = st.sidebar.selectbox("Select X-axis Feature", df.columns)
    feature_y = st.sidebar.selectbox("Select Y-axis Feature", df.columns)
    
    st.write(f"### Scatter Plot: {feature_x} vs {feature_y}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature_x], y=df[feature_y], ax=ax)
    st.pyplot(fig)
