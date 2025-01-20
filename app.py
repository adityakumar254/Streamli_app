import streamlit as st
import joblib
import pandas as pd

def load_model_and_encoder():
    model = joblib.load(r"E:\Project\Animal_health_classification\random_forest_model.pkl")
    encoder = joblib.load(r"E:\Project\Animal_health_classification\symptoms_encoder.pkl")
    return model, encoder

def predict(input_symptoms):
    model, encoder = load_model_and_encoder()
    symptom_columns = ["symptoms1", "symptoms2", "symptoms3", "symptoms4", "symptoms5"]
    input_df = pd.DataFrame([input_symptoms], columns=symptom_columns)
    input_features = encoder.transform(input_df)
    prediction = model.predict(input_features)
    return "Yes" if prediction[0] == 1 else "No"

# Streamlit UI with enhancements
st.set_page_config(page_title="Animal Health Classification", page_icon="🐾", layout="centered")

st.title("🐾 Animal Health Classification 🏥")
st.markdown("### Enter the symptoms to check if the condition is dangerous.")


# Input fields with sidebar
st.sidebar.header("Input Symptoms")
symptom1 = st.sidebar.text_input("Symptom 1")
symptom2 = st.sidebar.text_input("Symptom 2")
symptom3 = st.sidebar.text_input("Symptom 3")
symptom4 = st.sidebar.text_input("Symptom 4")
symptom5 = st.sidebar.text_input("Symptom 5")


# Prediction button
if st.sidebar.button("Predict"):
    if all([symptom1, symptom2, symptom3, symptom4, symptom5]):
        input_symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
        with st.spinner("Analyzing symptoms..."):
            result = predict(input_symptoms)
        if result == "Yes":
            st.error("🚨 The condition is dangerous!")
        else:
            st.success("✅ The condition is not dangerous!")
            st.balloons()
    else:
        st.warning("⚠️ Please fill all symptom fields.")

# Sidebar enhancements
st.sidebar.image(
    "https://s.tmimgcdn.com/scr/1204x1146/157100/animal-health-logo-template_157138-original.png",
    caption="Stay informed, stay safe!",
    use_container_width=True,  # Updated parameter
)


# Additional sections for interactivity
st.video("https://www.youtube.com/watch?v=OX29UgKQqOo&t=00s")
st.write("### Tips for Maintaining Animal Health")
st.markdown(
    """
    - 🥦 Provide proper nutrition
    - 🏥 Ensure regular veterinary checkups
    - 🧼 Maintain a clean living environment
    - 👀 Monitor symptoms early
    """
)
st.markdown("**Developed using Streamlit for an interactive experience.**")
st.write("\n")
st.write("\n")

# About Developer Section
with st.expander("📜 About the Developer"):
    st.markdown(
        """
        **Aditya Kumar Chauhan**  
        - **Profession**: Aspiring Data Analytics Professional with strong skills in data manipulation, statistical analysis, and programming.  
        - **Contact**:  
          📧 Email: [adityakumarchauhan87@gmail.com](mailto:adityakumarchauhan87@gmail.com)  
          💼 LinkedIn: [Aditya Kumar Chauhan](https://www.linkedin.com/in/aditya-chauhan09/)  
          🌐 GitHub: [adityakumar254](https://github.com/adityakumar254)  

        **Key Projects**  
        - **Animal Health Classification**: Applied machine learning for health classification tasks, including data preprocessing, model training, and evaluation.  
        - **Crop Disease Prediction**: Leveraged CNNs and TensorFlow for disease prediction, with deployment on AWS (in progress).  
        - **Currency Converter Chatbot**: Created a chatbot using Google Dialogflow, deployed on Telegram.  

        "I am passionate about leveraging data to solve real-world problems and am committed to creating impactful, data-driven solutions."  
        """
    )

st.markdown(
    """
   <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 5px 0;
        font-size: 12px;
        box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.2);
    }
    </style>
    <div class="corner-tag">
        Deployed by: Aditya Chauhan
    </div>
    """,
    unsafe_allow_html=True,
)
