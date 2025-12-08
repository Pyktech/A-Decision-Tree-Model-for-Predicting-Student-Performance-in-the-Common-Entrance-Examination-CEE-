import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import base64
import os

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_image(image_file):
    if os.path.exists(image_file):
        bin_str = get_base64_of_bin_file(image_file)
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        return True
    else:
        page_bg_img = '''
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        return False

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: rgba(0, 0, 30, 0.85);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: white;
        text-align: center;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-card {
        background: rgba(0, 20, 40, 0.85);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: white;
        text-align: center;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metrics-container {
        background-color: rgba(0, 10, 30, 0.85);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-top: 1rem;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .feature-card {
        background: rgba(0, 30, 60, 0.85);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        font-weight: bold;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        padding: 0.9rem 1.8rem;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 114, 255, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0, 114, 255, 0.6);
        background: linear-gradient(135deg, #00d2ff 0%, #0080ff 100%);
    }
    
    .sidebar .sidebar-content {
        background: rgba(0, 10, 30, 0.9);
        color: white;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Style for selectboxes and inputs in sidebar */
    .sidebar .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .sidebar .stSelectbox > div > div:hover {
        border: 1px solid rgba(0, 198, 255, 0.6) !important;
    }
    
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .info-text {
        color: #a0d2ff;
        font-style: italic;
    }
    
    /* Semi-transparent overlay for better readability */
    .main-content {
        background-color: rgba(0, 15, 40, 0.85);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Style for markdown text */
    p, li {
        color: #e0f0ff;
    }
    
    /* Style for success messages */
    .stAlert.success {
        background-color: rgba(0, 100, 0, 0.7) !important;
        color: white !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 0, 0.3);
    }
    
    /* Style for info messages */
    .stAlert.info {
        background-color: rgba(30, 144, 255, 0.7) !important;
        color: white !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(135, 206, 250, 0.5);
    }
    
    /* Style for warning messages */
    .stAlert.warning {
        background-color: rgba(255, 165, 0, 0.7) !important;
        color: white !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 215, 0, 0.5);
    }
    
    /* Professional icons using Unicode symbols */
    .icon-title::before {
        content: "📋";
        margin-right: 10px;
    }
    
    .icon-personal::before {
        content: "👤";
        margin-right: 10px;
    }
    
    .icon-academic::before {
        content: "📊";
        margin-right: 10px;
    }
    
    .icon-parental::before {
        content: "👨‍👩‍👧‍👦";
        margin-right: 10px;
    }
    
    .icon-result::before {
        content: "✅";
        margin-right: 10px;
    }
    
    .icon-metrics::before {
        content: "📈";
        margin-right: 10px;
    }
    
    .icon-feature::before {
        content: "🔑";
        margin-right: 10px;
    }
    
    .icon-insights::before {
        content: "💡";
        margin-right: 10px;
    }
    
    .icon-welcome::before {
        content: "👋";
        margin-right: 10px;
    }
    
    .icon-features::before {
        content: "📚";
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

bg_image_path = "coolbackgrounds-particles-stellar.png"
if os.path.exists(bg_image_path):
    set_bg_image(bg_image_path)
else:
    st.warning(f"Background image '{bg_image_path}' not found. Using default gradient.")

st.markdown("""
<div class="main-header">
    <h1>Student Performance Predictor (CEE)</h1>
    <p style="font-size: 1.3rem;">Predict student performance using a trained Decision Tree Classifier</p>
    <p>Select student attributes, then click Predict to get the result and evaluation metrics.</p>
</div>
""", unsafe_allow_html=True)

with open("decision_tree_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

st.sidebar.markdown("""
<div style="background: rgba(0, 10, 30, 0.9); padding: 1.2rem; border-radius: 15px; color: white; text-align: center; backdrop-filter: blur(15px); border: 1px solid rgba(255, 255, 255, 0.1);">
    <h3 class="icon-title">Student Input Features</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

def user_input_features():
    st.sidebar.markdown("### Personal Information")
    
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    time = st.sidebar.selectbox("Time available for study", ["One", "Two", "Three"])
    medium = st.sidebar.selectbox("Medium of Instruction", ["English", "Hindi", "Other"])
    
    st.sidebar.markdown("### Academic Performance")
    
    class_X_percentage = st.sidebar.selectbox("Class X Performance", ["Excellent", "Very Good", "Good", "Average"])
    class_XII_percentage = st.sidebar.selectbox("Class XII Performance", ["Excellent", "Very Good", "Good", "Average"])
    
    st.sidebar.markdown("### Parental Information")
    
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

predict_button = st.sidebar.button("Predict Performance", use_container_width=True)

if predict_button:

    with st.spinner('Analyzing student data...'):
        import time
        time.sleep(1)
        
        for col in le_dict:
            if col in sample_input.columns:
                sample_input = safe_transform(col, le_dict[col], sample_input)

        pred = clf.predict(sample_input)[0]

        num_to_code = {i: code for i, code in enumerate(le_dict["Performance"].classes_)}
        model_code = num_to_code.get(pred, "Vg")  
        prediction_map = {"Ex":"Excellent", "Vg":"Very Good", "Gd":"Good", "Av":"Average"}
        pred_readable = prediction_map.get(model_code, model_code)

        st.markdown(f"""
        <div class="prediction-card">
            <h2 class="icon-result">Prediction Result</h2>
            <h1 style="font-size: 2.8rem; margin: 1.2rem 0; color: #00c6ff; text-shadow: 0 0 15px rgba(0, 198, 255, 0.7);">{pred_readable}</h1>
            <p style="font-size: 1.3rem;">Predicted Student Performance</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Model Evaluation Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            st.markdown("**Training Set Metrics**")
            st.markdown("<span style='color:#00ffaa; font-size: 1.3rem;'>Accuracy: 0.6931</span>", unsafe_allow_html=True)
            st.markdown("F1-Score: 0.6926")
            st.markdown("RMSE: 1.0379")
            st.markdown("MAE: 0.5322")
            st.markdown("RAE: 0.5453")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            st.markdown("**Test Set Metrics**")
            st.markdown("<span style='color:#00aaff; font-size: 1.3rem;'>Accuracy: 0.6650</span>", unsafe_allow_html=True)
            st.markdown("F1-Score: 0.6632")
            st.markdown("RMSE: 1.0392")
            st.markdown("MAE: 0.5500")
            st.markdown("RAE: 0.5373")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Feature Importance Analysis")
        feature_importances = pd.DataFrame({
            'feature': clf.feature_names_in_,
            'importance': clf.feature_importances_
        }).sort_values(by='importance', ascending=False)

        st.markdown("#### Top Influential Factors")
        top_features = feature_importances.head(3)
        cols = st.columns(3)
        for idx, (col, row) in enumerate(zip(cols, top_features.itertuples())):
            with col:
                st.markdown(f"""
                <div class="feature-card">
                    <h4>{row.feature}</h4>
                    <p style="font-size: 1.7rem; color: #00c6ff;">{row.importance:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

        fig = px.bar(
            feature_importances,
            x='feature',
            y='importance',
            title="Feature Importance of Decision Tree Model",
            color='importance',
            color_continuous_scale=['#0072ff', '#00c6ff']
        )
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Importance Score",
            title_x=0.5,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Insights")
        st.info(f"""
        Based on the model analysis:
        - The most important factor affecting student performance is **{feature_importances.iloc[0]['feature']}**
        - The model has a training accuracy of **69.31%** and test accuracy of **66.50%**
        - This prediction should be used as a guide alongside other educational assessments
        """)

else:
    st.markdown("""
    <div style="text-align: center; padding: 2.5rem; background-color: rgba(0, 15, 40, 0.85); border-radius: 15px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1);">
        <h2 class="icon-welcome">Welcome to the Student Performance Predictor</h2>
        <p style="font-size: 1.3rem; color: #a0d2ff;">
            Please select the student attributes in the sidebar and click "Predict Performance" 
            to get started with the prediction.
        </p>
        <p style="color: #80c0ff; font-size: 1.1rem;">
            This tool uses a Decision Tree Classifier trained on historical student data to predict 
            performance in the Common Entrance Examination (CEE).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Features Used for Prediction")
    features = [
        "Gender",
        "Time spent studying",
        "Medium of instruction",
        "Class X Percentage",
        "Class XII Percentage",
        "Father's occupation",
        "Mother's occupation"
    ]
    
    cols = st.columns(2)
    for idx, feature in enumerate(features):
        with cols[idx % 2]:
            st.markdown(f'<div class="feature-card"><h4>{feature}</h4></div>', unsafe_allow_html=True)