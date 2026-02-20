import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Prediction (Logistic Regression)")

st.write("Enter passenger details to predict survival:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=29)
fare = st.number_input("Fare", min_value=0.0, value=80.0)
Siblings_Spouse= st.number_input("Siblings_Spouse", min_value=0, value=0)
Parents_Children=st.number_input("Parents_Children", min_value=0, value=0)
family_size=Siblings_Spouse+Parents_Children+1
sex = st.selectbox("Sex", ["Female", "Male"])
embarked = st.selectbox("Embarked", ["C", "Q", "S"])
title = st.selectbox("Title", ["Master", "Miss", "Mr", "Mrs", "Rare"])

# Encoding features manually
sex_female, sex_male = (1, 0) if sex == "Female" else (0, 1)
embarked_C, embarked_Q, embarked_S = 0, 0, 0
if embarked == "C":
    embarked_C = 1
elif embarked == "Q":
    embarked_Q = 1
else:
    embarked_S = 1

title_Master, title_Miss, title_Mr, title_Mrs, title_Rare = 0, 0, 0, 0, 0
if title == "Master":
    title_Master = 1
elif title == "Miss":
    title_Miss = 1
elif title == "Mr":
    title_Mr = 1
elif title == "Mrs":
    title_Mrs = 1
else:
    title_Rare = 1

# Creating dataframe for prediction
input_data = pd.DataFrame([{
    "Age": age,
    "Fare": fare,
    "Family_size": family_size,
    "Sex_female": sex_female,
    "Sex_male": sex_male,
    "Embarked_C": embarked_C,
    "Embarked_Q": embarked_Q,
    "Embarked_S": embarked_S,
    "Title_Master": title_Master,
    "Title_Miss": title_Miss,
    "Title_Mr": title_Mr,
    "Title_Mrs": title_Mrs,
    "Title_Rare": title_Rare,
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f" Survived (Probability: {prob:.2f})")
    else:
        st.error(f"Did not survive (Probability: {prob:.2f})")
