import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor (CEE)")
st.markdown("""
Predict student performance using a trained **Decision Tree Classifier**.  
Select student attributes, then click **Predict** to get the result and evaluation metrics.
""")

with open("decision_tree_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

st.sidebar.header("Student Input Features")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    time = st.sidebar.selectbox("Time available for study", ["One", "Two", "Three"])
    medium = st.sidebar.selectbox("Medium of Instruction", ["English", "Hindi", "Other"])
    class_X_percentage = st.sidebar.selectbox("Class X Performance", ["Excellent", "Very Good", "Good", "Average"])
    class_XII_percentage = st.sidebar.selectbox("Class XII Performance", ["Excellent", "Very Good", "Good", "Average"])
    father_occ = st.sidebar.selectbox("Father's Occupation", ["Doctor", "Engineer", "Others"])
    mother_occ = st.sidebar.selectbox("Mother's Occupation", ["Doctor", "Engineer", "Others"])
    
    mapping_perf = {"Excellent":"Ex", "Very Good":"Vg", "Good":"Gd", "Average":"Av"}
    
    data = {
        "Gender": gender.lower(),
        "time": time,
        "medium": medium,
        "Class_ X_Percentage": mapping_perf[class_X_percentage],
        "Class_XII_Percentage": mapping_perf[class_XII_percentage],
        "Father_occupation": father_occ.upper(),
        "Mother_occupation": mother_occ.upper()
    }
    
    return pd.DataFrame([data])

sample_input = user_input_features()

def safe_transform(col, encoder, df):
    val = df[col].iloc[0]
    if val in encoder.classes_:
        df[col] = encoder.transform([val])
    else:
        df[col] = encoder.transform([encoder.classes_[0]])
    return df

if st.sidebar.button("Predict"):

    for col in le_dict:
        if col in sample_input.columns:
            sample_input = safe_transform(col, le_dict[col], sample_input)

    pred = clf.predict(sample_input)[0]

    num_to_code = {i: code for i, code in enumerate(le_dict["Performance"].classes_)}
    model_code = num_to_code.get(pred, "Vg")  
    prediction_map = {"Ex":"Excellent", "Vg":"Very Good", "Gd":"Good", "Av":"Average"}
    pred_readable = prediction_map.get(model_code, model_code)

    st.subheader("🔹 Prediction Result")
    st.success(f"Predicted Performance: **{pred_readable}**")

    st.subheader("📊 Model Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Training Set Metrics**")
        st.markdown("<span style='color:green'>Accuracy: 0.6931</span>", unsafe_allow_html=True)
        st.markdown("F1-Score: 0.6926")
        st.markdown("RMSE: 1.0379")
        st.markdown("MAE: 0.5322")
        st.markdown("RAE: 0.5453")
    with col2:
        st.markdown("**Test Set Metrics**")
        st.markdown("<span style='color:blue'>Accuracy: 0.6650</span>", unsafe_allow_html=True)
        st.markdown("F1-Score: 0.6632")
        st.markdown("RMSE: 1.0392")
        st.markdown("MAE: 0.5500")
        st.markdown("RAE: 0.5373")

    st.subheader("📈 Feature Importance")
    feature_importances = pd.DataFrame({
        'feature': clf.feature_names_in_,
        'importance': clf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    fig = px.bar(
        feature_importances,
        x='feature',
        y='importance',
        title="Feature Importance of Decision Tree Model",
        color='importance',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

