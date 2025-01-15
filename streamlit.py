import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def preprocess_data(df):
    """Preprocess data by encoding categorical variables and dropping non-numeric columns."""
    # Drop non-numeric columns except the target variable
    df = df.select_dtypes(include=[np.number])
    return df

def calculate_metrics(model, X, y):
    """Calculate R² and RMSE for a given model."""
    predictions = model.predict(X)
    r2 = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return r2, rmse

def align_features(test_data, train_columns):
    """Align test data features with training data columns."""
    for col in train_columns:
        if col not in test_data.columns:
            test_data[col] = 0
    return test_data[train_columns]

def main():
    st.title("Bike Sharing Demand Prediction")

    # Upload training data
    st.header("Upload Training Data")
    train_file = st.file_uploader("Upload your training CSV file", type="csv", key="train")

    if train_file is not None:
        train_data = pd.read_csv(train_file)
        st.write("Training Data Preview:", train_data.head())

        # Preprocess data
        train_data = preprocess_data(train_data)

        # Split data into features and target
        if 'cnt' in train_data.columns:
            y_train = train_data['cnt']
            X_train = train_data.drop(['cnt'], axis=1)

            # Train model
            regmodel = LinearRegression()
            regmodel.fit(X_train, y_train)

            # Save feature names for alignment
            train_columns = X_train.columns

            # Calculate training metrics
            train_r2, train_rmse = calculate_metrics(regmodel, X_train, y_train)

            st.write(f"**Training Metrics:**")
            st.write(f"- R² Score: {train_r2:.4f}")
            st.write(f"- RMSE: {train_rmse:.4f}")
        else:
            st.error("The training data must include the 'cnt' column as the dependent variable.")

    # Upload test data
    st.header("Upload Test Data")
    test_file = st.file_uploader("Upload your test CSV file", type="csv", key="test")

    if test_file is not None:
        test_data = pd.read_csv(test_file)
        st.write("Test Data Preview:", test_data.head())

        # Preprocess data
        test_data = preprocess_data(test_data)

        if 'cnt' in test_data.columns:
            y_test = test_data['cnt']
            X_test = test_data.drop(['cnt'], axis=1)

            if train_file is not None:
                # Align test features with training features
                X_test = align_features(X_test, train_columns)

                # Calculate test metrics
                test_rmse = np.sqrt(mean_squared_error(y_test, regmodel.predict(X_test)))

                st.write(f"**Test Metrics:**")
                st.write(f"- RMSE: {test_rmse:.4f}")
            else:
                st.error("Please upload and train the model with training data first.")
        else:
            st.error("The test data must include the 'cnt' column as the dependent variable.")

if __name__ == "__main__":
    main()
