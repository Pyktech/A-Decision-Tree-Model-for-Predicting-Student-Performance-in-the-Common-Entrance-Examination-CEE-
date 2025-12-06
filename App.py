import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

# --- Load trained model and label encoders ---
with open("decision_tree_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

# --- Mapping for friendly labels ---
gender_map = {"Male": "male", "Female": "female"}
time_map = {"One": "ONE", "Two": "TWO", "Three": "THREE", "Four": "FOUR"}
medium_map = {"English": "ENGLISH", "Others": "OTHERS"}
class_x_map = {"Excellent": "Excellent", "Very Good": "Vg", "Good": "Gd", "Average": "Av", "Poor": "Pr"}
class_xii_map = {"Excellent": "Excellent", "Very Good": "Vg", "Good": "Gd", "Average": "Av", "Poor": "Pr"}
father_occ_map = {"Doctor": "DOCTOR", "Engineer": "ENGINEER", "Teacher": "TEACHER", "Others": "OTHERS"}
mother_occ_map = {"Doctor": "DOCTOR", "Engineer": "ENGINEER", "Teacher": "TEACHER", "Others": "OTHERS"}

# Helper function to map friendly input to code
def map_input(display_value, mapping_dict):
    return mapping_dict[display_value]

# --- App Layout ---
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.markdown("<h1 style='color: darkblue; text-align: center;'>CEE Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict a student's performance based on demographic and academic features</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input Section ---
st.header("Student Details")
col1, col2, col3 = st.columns(3)

with col1:
    gender_display = st.selectbox("Gender", list(gender_map.keys()))
    gender = map_input(gender_display, gender_map)
    time_display = st.selectbox("Time spent studying", list(time_map.keys()))
    time = map_input(time_display, time_map)
    medium_display = st.selectbox("Medium of instruction", list(medium_map.keys()))
    medium = map_input(medium_display, medium_map)

with col2:
    class_x_display = st.selectbox("Class X Performance", list(class_x_map.keys()))
    class_x = map_input(class_x_display, class_x_map)
    class_xii_display = st.selectbox("Class XII Performance", list(class_xii_map.keys()))
    class_xii = map_input(class_xii_display, class_xii_map)
    father_occ_display = st.selectbox("Father's Occupation", list(father_occ_map.keys()))
    father_occ = map_input(father_occ_display, father_occ_map)

with col3:
    mother_occ_display = st.selectbox("Mother's Occupation", list(mother_occ_map.keys()))
    mother_occ = map_input(mother_occ_display, mother_occ_map)

# --- Prepare input for prediction ---
sample_input = pd.DataFrame([{
    "Gender": gender,
    "time": time,
    "medium": medium,
    "Class_ X_Percentage": class_x,
    "Class_XII_Percentage": class_xii,
    "Father_occupation": father_occ,
    "Mother_occupation": mother_occ
}])

# Encode categorical features
for col in sample_input.columns:
    if col in le_dict:
        sample_input[col] = le_dict[col].transform(sample_input[col])

# --- Prediction Section ---
if st.button("Predict Performance"):
    pred = clf.predict(sample_input)
    pred_label = le_dict["Performance"].inverse_transform(pred)[0]

    st.subheader("Predicted Performance")
    st.markdown(f"<h2 style='color: green;'>{pred_label}</h2>", unsafe_allow_html=True)

    # Prediction probabilities
    if hasattr(clf, "predict_proba"):
        pred_prob = clf.predict_proba(sample_input)
        prob_df = pd.DataFrame(pred_prob, columns=le_dict["Performance"].classes_)
        st.subheader("Prediction Probabilities")
        st.dataframe(prob_df.T)

    # --- Evaluation Metrics (demo with training set if available) ---
    st.subheader("Model Evaluation Metrics")
    try:
        X_train = pd.read_csv("X_train.csv")  # optional
        y_train = pd.read_csv("y_train.csv")
        y_pred = clf.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred, average='weighted')
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        mae = mean_absolute_error(y_train, y_pred)
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write(f"**F1-Score:** {f1:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
    except:
        st.write("Evaluation metrics will appear if training/test data is provided.")

    # --- Feature Importance using Plotly ---
    if hasattr(clf, "feature_importances_"):
        st.subheader("Feature Importance")
        feat_imp = pd.Series(clf.feature_importances_, index=sample_input.columns)
        fig = px.bar(feat_imp.sort_values(ascending=True), orientation="h",
                     labels={"index": "Feature", "value": "Importance"},
                     color=feat_imp.sort_values(ascending=True))
        st.plotly_chart(fig)

