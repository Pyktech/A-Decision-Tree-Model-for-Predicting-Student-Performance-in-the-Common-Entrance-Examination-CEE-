import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

# --- Load trained model and label encoders ---
with open("decision_tree_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

# --- App Title ---
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.markdown("<h1 style='color: darkblue; text-align: center;'>CEE Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict a student's performance based on demographic and academic features</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input Section ---
st.header("Student Details")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", le_dict["Gender"].classes_)
    time = st.selectbox("Time spent studying", le_dict["time"].classes_)
    medium = st.selectbox("Medium of instruction", le_dict["medium"].classes_)

with col2:
    class_x = st.selectbox("Class X Percentage", le_dict["Class_ X_Percentage"].classes_)
    class_xii = st.selectbox("Class XII Percentage", le_dict["Class_XII_Percentage"].classes_)
    father_occ = st.selectbox("Father's Occupation", le_dict["Father_occupation"].classes_)

with col3:
    mother_occ = st.selectbox("Mother's Occupation", le_dict["Mother_occupation"].classes_)

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

# --- Prediction Button ---
if st.button("Predict Performance"):
    # Predict
    pred = clf.predict(sample_input)
    pred_label = le_dict["Performance"].inverse_transform(pred)[0]

    # Prediction probabilities if available
    if hasattr(clf, "predict_proba"):
        pred_prob = clf.predict_proba(sample_input)
        prob_df = pd.DataFrame(pred_prob, columns=le_dict["Performance"].classes_)
        st.subheader("Prediction Probabilities")
        st.dataframe(prob_df.T)

    # Display prediction
    st.subheader("Predicted Performance")
    st.markdown(f"<h2 style='color: green;'>{pred_label}</h2>", unsafe_allow_html=True)

    # --- Evaluation Metrics ---
    st.subheader("Model Evaluation Metrics")

    # For demonstration: we can show metrics on the training set (replace with test set if available)
    # Load X_train, y_train if available for real metrics
    try:
        X_train = pd.read_csv("X_train.csv")  # optional, remove if not using
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
        st.write("Evaluation metrics will be displayed when test/training data is provided.")

    # --- Feature Importance ---
    if hasattr(clf, "feature_importances_"):
        import matplotlib.pyplot as plt
        st.subheader("Feature Importance")
        feat_imp = pd.Series(clf.feature_importances_, index=sample_input.columns)
        fig, ax = plt.subplots()
        feat_imp.sort_values(ascending=True).plot(kind="barh", ax=ax, color="skyblue")
        ax.set_xlabel("Importance")
        st.pyplot(fig)

