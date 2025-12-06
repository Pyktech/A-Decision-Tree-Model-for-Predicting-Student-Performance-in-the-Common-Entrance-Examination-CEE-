import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon=":mortar_board:",
    layout="wide"
)

st.title("🎓 Student Performance Predictor (CEE)")
st.markdown("""
Predict student performance using a trained **Decision Tree Classifier**.  
Enter student attributes below and get the predicted performance along with evaluation metrics.
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
    class_X_percentage = st.sidebar.selectbox("Class X Performance", ["Very Good", "Excellent", "Good"])
    class_XII_percentage = st.sidebar.selectbox("Class XII Performance", ["Very Good", "Excellent", "Good"])
    father_occ = st.sidebar.selectbox("Father's Occupation", ["Doctor", "Engineer", "Others"])
    mother_occ = st.sidebar.selectbox("Mother's Occupation", ["Doctor", "Engineer", "Others"])
    
    mapping_perf = {"Very Good":"Vg", "Excellent":"Ex", "Good":"Gd"}
    
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

for col in le_dict:
    if col in sample_input.columns:
        sample_input = safe_transform(col, le_dict[col], sample_input)

pred = clf.predict(sample_input)[0]
prediction_map = {"Ex":"Excellent", "Vg":"Very Good", "Gd":"Good"}
pred_readable = prediction_map.get(pred, pred)

st.subheader("🔹 Prediction Result")
st.success(f"Predicted Performance: **{pred_readable}**")

st.subheader("📊 Model Evaluation Metrics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Training Set Metrics**")
    st.write(f"Accuracy: 0.6931")
    st.write(f"F1-Score: 0.6926")
    st.write(f"RMSE: 1.0379")
    st.write(f"MAE: 0.5322")
    st.write(f"RAE: 0.5453")

with col2:
    st.markdown("**Test Set Metrics**")
    st.write(f"Accuracy: 0.6650")
    st.write(f"F1-Score: 0.6632")
    st.write(f"RMSE: 1.0392")
    st.write(f"MAE: 0.5500")
    st.write(f"RAE: 0.5373")

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
