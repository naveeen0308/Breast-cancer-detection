import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
import seaborn as sns

# Import necessary libraries
import matplotlib.pyplot as plt

# Load the breast cancer dataset
data = load_breast_cancer()

# Create a Pandas DataFrame from the data
df = pd.DataFrame(data.data, columns=data.feature_names)

# Extract features (X) and labels (y)
X = data.data
y = data.target  # 0 = malignant, 1 = benign

# Map target labels for clarity: 0 -> malignant, 1 -> benign
label_map = {0: "Malignant", 1: "Benign"}

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the AdaBoost classifier
model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit app title
st.title("Breast Cancer Detection")

# Sidebar for user input features
st.sidebar.header("Input Features")

# Function to get user input
def user_input_features():
    features = {}
    for feature in data.feature_names:
        features[feature] = st.sidebar.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    return pd.DataFrame(features, index=[0])

# Get user input
input_df = user_input_features()

# Standardize the user input features
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display the prediction
st.subheader("Prediction")
st.write(f"The model predicts that the tumor is **{label_map[prediction[0]]}**")

# Display the prediction probability
st.subheader("Prediction Probability")
st.write(prediction_proba)

# Display model results
st.subheader("Model Results")
st.write(f"Model Accuracy: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(pd.DataFrame(conf_matrix,
                      index=["Actual Malignant", "Actual Benign"],
                      columns=["Predicted Malignant", "Predicted Benign"]))
st.write("\nClassification Report:")
st.text(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

# Display feature importance
st.subheader("Feature Importance")
feature_importance = pd.Series(model.feature_importances_, index=data.feature_names).sort_values(ascending=False)
st.bar_chart(feature_importance)

# Display correlation heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)

# Display the actual vs predicted values in a table with enhanced visuals
st.subheader("Actual vs Predicted Values")

# Create a DataFrame for actual vs predicted values
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df['Actual'] = results_df['Actual'].map(label_map)
results_df['Predicted'] = results_df['Predicted'].map(label_map)

# Add a status column to indicate if the prediction is correct
results_df['Status'] = np.where(results_df['Actual'] == results_df['Predicted'], 'Correct', 'Incorrect')

# Define a function to color the rows based on the status
def color_status(val):
    color = 'green' if val == 'Correct' else 'red'
    return f'background-color: {color}'

# Apply the color function to the DataFrame
styled_results_df = results_df.style.applymap(color_status, subset=['Status'])

# Display the styled DataFrame
st.dataframe(styled_results_df)
