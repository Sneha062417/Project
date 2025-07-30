import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle # Added pickle module
from scipy import stats # Added scipy.stats module

# Load the data
@st.cache_data
def load_data():
   
    data = pd.read_excel('shopping_trends.xlsx')
    return data

data = load_data()

# Set page title and layout
st.set_page_config(page_title="Shopping Trends Analysis", layout="wide")
st.title("Shopping Trends Analysis with Linear Regression")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Analysis:",
                           ["Data Overview", "Exploratory Analysis",
                            "Purchase Amount Prediction", "Review Rating Prediction",
                            "Previous Purchases Prediction"])

# Data Overview
if options == "Data Overview":
    st.header("Dataset Overview")

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.write(data)

    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Data information
    st.subheader("Data Information")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")
    st.write("Columns and their data types:")
    st.write(data.dtypes)

    # Missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

# Exploratory Analysis
elif options == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")

    # Create two columns for plots
    col1, col2 = st.columns(2)

    with col1:
        # Plot 1: Age Distribution
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['Age'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

        # Plot 2: Purchase Amount by Category
        st.subheader("Purchase Amount by Category")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Category', y='Purchase Amount (USD)', data=data, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Plot 3: Payment Method Distribution
        st.subheader("Payment Method Distribution")
        fig, ax = plt.subplots()
        data['Payment Method'].value_counts().plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Plot 4: Review Rating Distribution
        st.subheader("Review Rating Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['Review Rating'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

        # Plot 5: Previous Purchases Distribution
        st.subheader("Previous Purchases Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['Previous Purchases'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        # Plot 6: Purchase Amount by Season
        st.subheader("Purchase Amount by Season")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Season', y='Purchase Amount (USD)', data=data, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Plot 7: Subscription Status Distribution
        st.subheader("Subscription Status Distribution")
        fig, ax = plt.subplots()
        data['Subscription Status'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

        # Plot 8: Shipping Type Distribution
        st.subheader("Shipping Type Distribution")
        fig, ax = plt.subplots()
        data['Shipping Type'].value_counts().plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Plot 9: Purchase Amount vs. Age
        st.subheader("Purchase Amount vs. Age")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Age', y='Purchase Amount (USD)', data=data, alpha=0.5, ax=ax)
        st.pyplot(fig)

        # Plot 10: Review Rating vs. Purchase Amount
        st.subheader("Review Rating vs. Purchase Amount")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Review Rating', y='Purchase Amount (USD)', data=data, alpha=0.5, ax=ax)
        st.pyplot(fig)

# Purchase Amount Prediction
elif options == "Purchase Amount Prediction":
    st.header("Predict Purchase Amount")

    # Prepare data for modeling
    X = data[['Age', 'Review Rating', 'Previous Purchases']]
    y = data['Purchase Amount (USD)']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"R-squared: {r2:.3f}")
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Show coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    st.write(coef_df)

    # Visualization
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=ax)
    st.pyplot(fig)

# Review Rating Prediction
elif options == "Review Rating Prediction":
    st.header("Predict Review Rating")

    # Prepare data for modeling
    X = data[['Age', 'Purchase Amount (USD)', 'Previous Purchases']]
    y = data['Review Rating']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"R-squared: {r2:.3f}")
    st.write(f"Mean Squared Error: {mse:.4f}")

    # Show coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    st.write(coef_df)

    # Visualization
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)

    # Residual plot
    st.subheader("Residual Plot")
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)

# Previous Purchases Prediction
elif options == "Previous Purchases Prediction":
    st.header("Predict Previous Purchases")

    # Prepare data for modeling
    X = data[['Age', 'Purchase Amount (USD)', 'Review Rating']]
    y = data['Previous Purchases']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"R-squared: {r2:.3f}")
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Show coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    st.write(coef_df)

    # Visualization
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)

    # Feature relationships
    st.subheader("Feature Relationships")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    sns.scatterplot(x='Age', y='Previous Purchases', data=data, ax=ax[0])
    ax[0].set_title('Age vs Previous Purchases')

    sns.scatterplot(x='Purchase Amount (USD)', y='Previous Purchases', data=data, ax=ax[1])
    ax[1].set_title('Purchase Amount vs Previous Purchases')

    sns.scatterplot(x='Review Rating', y='Previous Purchases', data=data, ax=ax[2])
    ax[2].set_title('Review Rating vs Previous Purchases')

    st.pyplot(fig)
