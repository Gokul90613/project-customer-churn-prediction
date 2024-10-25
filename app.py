# coding: utf-8

import pandas as pd
import pickle
import streamlit as st

# Load the training data to get the exact columns used during model training
df_1 = pd.read_csv("first_telc.csv")

# Load your model
model = pickle.load(open("model.sav", "rb"))

# Page title
st.title("Customer Churn Prediction")

# Collect user input for each feature
st.sidebar.header("Enter Customer Details:")
inputQuery1 = st.sidebar.selectbox("Senior Citizen", [0, 1])
inputQuery2 = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=0.1)
inputQuery3 = st.sidebar.number_input("Total Charges", min_value=0.0, step=0.1)
inputQuery4 = st.sidebar.selectbox("Partner", ["Yes", "No"])
inputQuery5 = st.sidebar.selectbox("Dependents", ["Yes", "No"])
inputQuery6 = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
inputQuery7 = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
inputQuery8 = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
inputQuery9 = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
inputQuery10 = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
inputQuery11 = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
inputQuery12 = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# Prepare the input data
data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, 
         inputQuery7, inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12]]
new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 
                                     'Partner', 'Dependents', 'MultipleLines', 'OnlineSecurity', 
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract', 
                                     'PaperlessBilling'])

# Combine new input with the original training data to get consistent dummy columns
df_2 = pd.concat([df_1, new_df], ignore_index=True) 

# Apply one-hot encoding
new_df_dummies = pd.get_dummies(df_2[['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 
                                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                      'TechSupport', 'Contract', 'PaperlessBilling']], drop_first=False)

# Remove duplicate columns
new_df_dummies = new_df_dummies.loc[:, ~new_df_dummies.columns.duplicated()]

# Align with model columns
new_df_dummies = new_df_dummies.reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction and probability
if st.sidebar.button("Predict Churn"):
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]
    
    # Display result
    if single == 1:
        st.write("### This customer is likely to churn!")
    else:
        st.write("### This customer is likely to continue!")
    
    st.write("Confidence: {:.2f}%".format(probability[0] * 100))
