import streamlit as st
import pandas as pd
import pickle

with open("decision_tree_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

st.title("CEE Student Performance Prediction")
st.write("Enter student details to predict performance:")

gender = st.selectbox("Gender", ["male", "female"])
time = st.selectbox("Time spent studying", ["ONE", "TWO", "THREE", "FOUR"])
medium = st.selectbox("Medium of instruction", ["ENGLISH", "OTHERS"])
class_x_perc = st.selectbox("Class X Percentage", ["Excellent", "Good", "Average", "Poor"])
class_xii_perc = st.selectbox("Class XII Percentage", ["Excellent", "Good", "Average", "Poor"])
father_occ = st.selectbox("Father's occupation", ["DOCTOR", "TEACHER", "OTHERS"])
mother_occ = st.selectbox("Mother's occupation", ["DOCTOR", "TEACHER", "OTHERS"])

sample_input = pd.DataFrame([{
    "Gender": gender,
    "time": time,
    "medium": medium,
    "Class_ X_Percentage": class_x_perc,
    "Class_XII_Percentage": class_xii_perc,
    "Father_occupation": father_occ,
    "Mother_occupation": mother_occ
}])

for col in sample_input.columns:
    if col in le_dict:
        sample_input[col] = le_dict[col].transform(sample_input[col])

if st.button("Predict Performance"):
    pred = clf.predict(sample_input)
    predicted_label = le_dict["Performance"].inverse_transform(pred)
    st.success(f"Predicted Performance: {predicted_label[0]}")
