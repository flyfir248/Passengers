Titanic Survival Prediction

This project demonstrates a machine learning workflow for predicting survival on the Titanic using a Random Forest Classifier. The project includes data preprocessing, model training, evaluation, and a web application built with Streamlit to make predictions based on user inputs. This project is part of the UNext Manipal Diploma Course.
Project Structure

bash

.
├── titanic.csv                # The dataset
├── titanic_model.pkl          # The trained model
├── train_model.py             # Script for training the model
└── app.py                     # Streamlit application
└── readme.md                  # Project documentation

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction

Install the required packages:

bash

    pip install pandas scikit-learn streamlit joblib

Data Preparation

The dataset used in this project is the Titanic dataset, which can be found at Kaggle Titanic Dataset. The dataset is stored in titanic.csv.
Training the Model

The train_model.py script preprocesses the data, trains a Random Forest Classifier, and saves the trained model to a file.

python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
titanic_data = pd.read_csv('titanic.csv')

# Preprocess the data
titanic_data.dropna(inplace=True)  # Drop rows with missing values for simplicity
X = titanic_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, 'titanic_model.pkl')

Running the Streamlit App

The app.py script provides a web interface for making predictions using the trained model.

python

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("titanic_model.pkl")

st.title('Titanic Survival Prediction')

# User inputs
pcls = st.select_slider('Choose passenger class', [1, 2, 3])
age = st.slider('Input Age', 0, 100)
sib = st.slider('Input siblings', 0, 10)
parch = st.slider('Input parents/children', 0, 2)
fare = st.number_input('Fare amount', 0, 100)

# Column names
column_names = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

def predict():
    row = np.array([pcls, age, sib, parch, fare])
    X = pd.DataFrame([row], columns=column_names)
    prediction = model.predict(X)[0]
    if prediction == 1:
        st.success('Passenger Survived')
    else:
        st.error('Passenger Did Not Survive')

st.button('Predict', on_click=predict)

To run the Streamlit app, use the following command:

bash

streamlit run app.py

Conclusion

This project demonstrates the application of a machine learning model in a real-world scenario, from data preprocessing to model deployment using Streamlit. It serves as a useful reference for implementing similar projects.

For further details, you can refer to the code in the repository.