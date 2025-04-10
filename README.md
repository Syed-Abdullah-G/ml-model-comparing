# 📊 Machine Learning Model Comparison

This application provides a comparative analysis of two machine learning models using different data preprocessing techniques. It is built using [Streamlit](https://streamlit.io/) and is accessible as a web app.

## 🔧 Features

- **Missing Data Handling**  
  The app supports two strategies for dealing with missing data:
  - **Mean Imputation**: Replaces null values with the mean of the respective column.
  - **Deletion**: Removes rows that contain null values entirely.

- **Categorical Data Processing**  
  - Uses **One-Hot Encoding** to convert categorical variables into numerical form.

- **Model Comparison**  
  The app trains the dataset on two popular regression models:
  - **Linear Regression**
  - **Random Forest Regressor**

- **Performance Evaluation**  
  - Computes the **Mean Squared Error (MSE)** for each combination of preprocessing strategy and model.
  - Identifies the **best-performing model and method** based on the lowest MSE.

## 🧠 Goal

To determine the optimal combination of data processing strategy and machine learning model for accurate predictive analysis.

## 🌐 Live Demo

Check out the app here:  
👉 [Model Comparison App](https://modelcomparing.streamlit.app/)

---

Feel free to contribute or provide feedback!
