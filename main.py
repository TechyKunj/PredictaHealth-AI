import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import json
import google.generativeai as genai
import os
from typing import Dict, List, Any
import time

# Page configuration
st.set_page_config(
    page_title="PredictaHealth AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini API


def initialize_gemini(api_key=None):
    """Initialize Gemini AI with API key"""
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Test the connection
        test_response = model.generate_content("Hello")
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None


# Initialize session state for API key
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

# Custom CSS (keeping the previous styling)

st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #1F3A93; /* Professional dark blue */
        margin-bottom: 2rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Disease Cards */
    .disease-card {
        background: linear-gradient(135deg, #4A90E2 0%, #50E3C2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        color: #fff;
        font-weight: 500;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .disease-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }

    /* AI Insight Boxes */
    .ai-insight {
        background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #00CED1;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        color: #fff;
    }

    /* Gemini Response */
    .gemini-response {
        background: linear-gradient(135deg, #7F7FD5 0%, #86A8E7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        font-weight: 500;
    }

    /* Prediction Results */
    .prediction-result {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .positive-result {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF4757 100%);
        color: white;
    }
    .negative-result {
        background: linear-gradient(135deg, #2ED573 0%, #1EAE60 100%);
        color: white;
    }

    /* Input Sections */
    .input-section {
        background: #F5F6FA; /* Soft light gray */
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4A90E2 0%, #50E3C2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.16);
    }

    /* Chat Messages */
    .chat-message {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 0.95rem;
    }
            
    .user-message {
        background: #D6EAF8; /* Slightly darker blue */
        color: #1B4F72;       /* Dark text for readability */
        border-left: 4px solid #1F618D;
    }
    .ai-message {
        background: #E8F8F5; /* Light teal background */
        color: #1A5276;      /* Darker text */
        border-left: 4px solid #148F77;
    }

</style>
""", unsafe_allow_html=True)

# Gemini AI Functions


class GeminiHealthAssistant:
    def __init__(self, model):
        self.model = model

    def get_health_insights(self, prediction_result: int, user_data: Dict, disease_type: str) -> str:
        """Generate personalized health insights using Gemini"""
        if not self.model:
            return "AI insights unavailable - Gemini API not configured"

        prompt = f"""
        As a medical AI assistant, analyze this health prediction result and provide comprehensive insights:
        
        Disease Type: {disease_type}
        Prediction Result: {'High Risk' if prediction_result == 1 else 'Low Risk'}
        User Health Data: {user_data}
        
        Please provide:
        1. Risk Assessment Analysis
        2. Key Risk Factors (based on the input data)
        3. Personalized Prevention Recommendations
        4. Lifestyle Modifications
        5. When to Seek Medical Attention
        6. Long-term Health Management Tips
        
        Format the response in clear sections with bullet points.
        Include disclaimers about consulting healthcare professionals.
        Keep it informative but not alarming.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def analyze_symptoms(self, symptoms: str) -> str:
        """Analyze symptoms using Gemini"""
        if not self.model:
            return "Symptom analysis unavailable - Gemini API not configured"

        prompt = f"""
        As a medical AI assistant, analyze these symptoms and provide insights:
        
        Symptoms: {symptoms}
        
        Please provide:
        1. Possible conditions that might be related
        2. Severity assessment
        3. Immediate care recommendations
        4. When to seek emergency care
        5. Questions to ask a healthcare provider
        
        Important: Always emphasize that this is not a medical diagnosis and professional consultation is required.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error analyzing symptoms: {str(e)}"

    def get_health_tips(self, disease_type: str, age: int, gender: str) -> str:
        """Get personalized health tips"""
        if not self.model:
            return "Health tips unavailable - Gemini API not configured"

        prompt = f"""
        Provide personalized health tips for preventing {disease_type} for a {age}-year-old {gender}:
        
        Include:
        1. Diet recommendations
        2. Exercise suggestions
        3. Lifestyle modifications
        4. Preventive measures
        5. Regular health checkups needed
        
        Make it practical and actionable.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating health tips: {str(e)}"

    def chat_with_health_assistant(self, user_message: str, conversation_history: List = None) -> str:
        """Chat interface with health assistant"""
        if not self.model:
            return "Chat unavailable - Gemini API not configured"

        context = "You are a helpful medical AI assistant. Provide accurate health information while always emphasizing the need for professional medical consultation for diagnosis and treatment."

        if conversation_history:
            # Last 3 messages for context
            context += f"\n\nConversation history: {conversation_history[-3:]}"

        prompt = f"{context}\n\nUser question: {user_message}\n\nPlease provide a helpful, accurate response:"

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error in chat: {str(e)}"


# Initialize Gemini assistant
gemini_assistant = GeminiHealthAssistant(st.session_state.gemini_model)

# Load models (keeping original function)


@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'diabetes': 'diabetes_model.sav',
        'heart': 'heart_disease_model.sav',
        'ckd': 'ckd_model.sav',
        'cancer': 'Cancer_model.sav',
        'parkinson': 'parkinson_model .sav'
    }

    for name, file in model_files.items():
        try:
            models[name] = pickle.load(open(file, 'rb'))
        except FileNotFoundError:
            st.warning(f"Model file {file} not found!")
            models[name] = None

    return models


models = load_models()

# Header
st.markdown('<h1 class="main-header">⚕️PredictaHealth AI</h1>',
            unsafe_allow_html=True)
st.markdown("### Advanced Multi-Disease Prediction Powered by Google Gemini AI")

# Sidebar navigation
with st.sidebar:
    st.markdown("### 🤖 Gemini AI Configuration")

    # API Key input
    api_key_input = st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        value=st.session_state.gemini_api_key,
        placeholder="Enter your API key here...",
        help="Get your free API key from https://makersuite.google.com/app/apikey"
    )

    # Connect/Disconnect buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Connect", key="connect_api"):
            if api_key_input.strip():
                with st.spinner("Testing connection..."):
                    model = initialize_gemini(api_key_input.strip())
                    if model:
                        st.session_state.gemini_api_key = api_key_input.strip()
                        st.session_state.gemini_model = model
                        st.success("Connected! ✅")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Connection failed! ❌")
            else:
                st.error("Please enter an API key!")

    with col2:
        if st.button("🔌 Disconnect", key="disconnect_api"):
            st.session_state.gemini_api_key = ""
            st.session_state.gemini_model = None
            st.info("Disconnected!")
            time.sleep(1)
            st.rerun()

    # AI Status
    st.markdown("### 🤖 AI Status")
    if st.session_state.gemini_model:
        st.success("Gemini AI: Connected ✅")
        st.info(f"Model: gemini-1.5-flash")
    else:
        st.error("Gemini AI: Not Connected ❌")

    # Quick setup guide
    with st.expander("📋 Quick Setup Guide", expanded=False):
        st.markdown("""
        **Step 1:** Get your free API key
        - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
        - Sign in with Google account
        - Create new API key
        
        **Step 2:** Enter key above and click Connect
        
        **Step 3:** Start using AI features!
        
        🔒 **Privacy:** Your API key is stored only in this session and is not saved permanently.
        """)

    st.markdown("---")
    st.markdown("### 🔍 Select Feature")
    selected = option_menu(
        'Health Assistant',
        ['🩺 Dashboard', '🤖 AI Chat', '🔍 Symptom Analyzer', '🍯 Diabetes',
            '❤️ Heart Disease', '🫘 Kidney Disease', '🫁 Cancer', '🧠 Parkinson\'s'],
        icons=['house', 'robot', 'search', 'droplet',
               'heart', 'kidneys', 'lungs', 'brain'],
        menu_icon='hospital',
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#2E86AB", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#e3f2fd", "color": "#333"},
            "nav-link-selected": {"background-color": "#2E86AB", "color": "white"},
        }
    )

# Dashboard
if selected == '🩺 Dashboard':
    st.markdown('<div class="disease-card"><h2>🎯 AI-Powered Health Dashboard</h2><p>Your comprehensive health prediction platform powered by Google Gemini AI</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏥 Available Predictions")
        predictions = ['Diabetes', 'Heart Disease',
                       'Kidney Disease', 'Cancer', 'Parkinson\'s']
        for pred in predictions:
            st.info(f"✅ {pred} Risk Assessment")

    with col2:
        st.markdown("### 🤖 AI Features")
        ai_features = [
            "🔍 Intelligent Symptom Analysis",
            "💬 24/7 Health Chat Assistant",
            "📊 Personalized Health Insights",
            "🎯 Risk Factor Analysis",
            "📋 Preventive Care Recommendations"
        ]
        for feature in ai_features:
            st.success(feature)

    if st.session_state.gemini_model:
        st.markdown("### 💡 Daily Health Tip")
        with st.spinner("Getting personalized health tip..."):
            health_tip = gemini_assistant.get_health_tips(
                "general wellness", 35, "adult")
            st.markdown(
                f'<div class="gemini-response">{health_tip}</div>', unsafe_allow_html=True)

# AI Chat Assistant
elif selected == '🤖 AI Chat':
    st.markdown('<div class="disease-card"><h2>🤖 AI Health Chat Assistant</h2><p>Chat with Gemini AI for personalized health guidance</p></div>', unsafe_allow_html=True)

    if not st.session_state.gemini_model:
        st.warning("🤖 Gemini AI is not connected!")
        st.info(
            "👆 Please enter your API key in the sidebar to enable AI chat features.")

        with st.expander("🔑 How to get Gemini API Key", expanded=True):
            st.markdown("""
            **Get your free Gemini API key in 3 steps:**
            1. 🌐 Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. 🔑 Sign in and create a new API key
            3. 📋 Copy the key and paste it in the sidebar
            
            **💰 Cost:** Free tier includes generous usage limits!
            **🔒 Privacy:** Your API key is only stored in this session.
            """)

        # Show demo conversation
        st.markdown("### 🎯 Demo: What you can ask the AI")
        demo_questions = [
            "What are the early signs of diabetes?",
            "How can I lower my blood pressure naturally?",
            "What exercises are good for heart health?",
            "Explain my BMI results",
            "What foods should I avoid for kidney health?"
        ]

        for question in demo_questions:
            st.info(f"💬 {question}")
    else:
        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat interface
        st.markdown("### 💬 Chat with Your AI Health Assistant")

        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="chat-message ai-message"><strong>🤖 AI Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)

        # Chat input
        user_input = st.text_input(
            "Ask me anything about health, symptoms, or medical conditions:", key="chat_input")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("Send Message", key="send_chat"):
                if user_input.strip():
                    # Add user message to history
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_input})

                    # Get AI response
                    with st.spinner("AI is thinking..."):
                        ai_response = gemini_assistant.chat_with_health_assistant(
                            user_input, st.session_state.chat_history)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": ai_response})

                    st.rerun()

        with col2:
            if st.button("Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

# Symptom Analyzer
elif selected == '🔍 Symptom Analyzer':
    st.markdown('<div class="disease-card"><h2>🔍 AI Symptom Analyzer</h2><p>Analyze your symptoms with Gemini AI</p></div>', unsafe_allow_html=True)

    if not st.session_state.gemini_model:
        st.warning("🤖 Gemini AI is not connected!")
        st.info(
            "👆 Please enter your API key in the sidebar to enable symptom analysis.")

        # Show example analysis
        st.markdown("### 🎯 Example: AI Symptom Analysis")
        with st.expander("See sample analysis", expanded=True):
            st.markdown("""
            **Sample Input:** "I have been experiencing headaches, fatigue, and difficulty sleeping for the past week"
            
            **AI Analysis would include:**
            - 🔍 Possible related conditions
            - ⚠️ Severity assessment  
            - 🏥 When to seek medical care
            - 💡 Self-care recommendations
            - ❓ Questions for your doctor
            """)
    else:
        st.markdown("### 📝 Describe Your Symptoms")

        symptoms = st.text_area("Please describe your symptoms in detail:",
                                placeholder="e.g., I have been experiencing headaches, fatigue, and difficulty sleeping for the past week...",
                                height=150)

        if st.button("🔍 Analyze Symptoms"):
            if symptoms.strip():
                with st.spinner("Analyzing your symptoms with AI..."):
                    analysis = gemini_assistant.analyze_symptoms(symptoms)
                    st.markdown(
                        f'<div class="gemini-response"><h4>🤖 AI Analysis Results</h4>{analysis}</div>', unsafe_allow_html=True)

                    st.warning("⚠️ **Important Disclaimer:** This analysis is for informational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.")
            else:
                st.error("Please describe your symptoms before analyzing.")

# Enhanced Diabetes Prediction with Gemini
elif selected == '🍯 Diabetes':
    st.markdown('<div class="disease-card"><h2>🍯 Diabetes Prediction with AI Insights</h2><p>Advanced diabetes risk assessment enhanced by Gemini AI</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Health Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input(
            'Number of Pregnancies', min_value=0, max_value=20, value=0)
        glucose = st.number_input(
            'Glucose Level (mg/dL)', min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input(
            'Blood Pressure (mm Hg)', min_value=0, max_value=200, value=80)

    with col2:
        skin_thickness = st.number_input(
            'Skin Thickness (mm)', min_value=0, max_value=100, value=20)
        insulin = st.number_input(
            'Insulin Level (μU/mL)', min_value=0, max_value=1000, value=80)
        bmi = st.number_input('BMI', min_value=0.0,
                              max_value=70.0, value=25.0, step=0.1)

    with col3:
        dpf = st.number_input('Diabetes Pedigree Function',
                              min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input('Age', min_value=1, max_value=120, value=30)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button('🔍 Predict Diabetes Risk with AI Analysis', key='diabetes_predict'):
        if models['diabetes'] is not None:
            user_input = [pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]

            try:
                prediction = models['diabetes'].predict([user_input])

                if prediction[0] == 1:
                    st.markdown(
                        '<div class="prediction-result positive-result">⚠️ HIGH DIABETES RISK DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="prediction-result negative-result">✅ LOW DIABETES RISK</div>', unsafe_allow_html=True)

                # Create user data dictionary for Gemini analysis
                user_data = {
                    'pregnancies': pregnancies,
                    'glucose': glucose,
                    'blood_pressure': blood_pressure,
                    'skin_thickness': skin_thickness,
                    'insulin': insulin,
                    'bmi': bmi,
                    'diabetes_pedigree_function': dpf,
                    'age': age
                }

                # Get Gemini AI insights
                if st.session_state.gemini_model:
                    with st.spinner("Generating personalized AI insights..."):
                        ai_insights = gemini_assistant.get_health_insights(
                            prediction[0], user_data, "diabetes")
                        st.markdown(
                            f'<div class="gemini-response"><h4>🤖 Gemini AI Health Insights</h4>{ai_insights}</div>', unsafe_allow_html=True)
                else:
                    st.warning(
                        "🤖 Connect Gemini AI in the sidebar for personalized insights!")
                    with st.expander("What AI insights would include:", expanded=False):
                        st.markdown("""
                        - 📊 **Detailed Risk Analysis** of your input values
                        - 🎯 **Key Risk Factors** identified from your data
                        - 💡 **Personalized Prevention Tips** based on your profile
                        - 🏃 **Lifestyle Modifications** specifically for you
                        - 🏥 **When to Seek Medical Care** guidance
                        - 📈 **Long-term Health Management** strategies
                        """)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.error("Diabetes model not available!")

# Enhanced Heart Disease Prediction with Gemini
elif selected == '❤️ Heart Disease':
    st.markdown('<div class="disease-card"><h2>❤️ Heart Disease Prediction with AI</h2><p>Cardiovascular risk assessment enhanced by Gemini AI</p></div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About Heart Disease Prediction", expanded=False):
        st.write("""
        This model analyzes 13 cardiovascular indicators to predict heart disease risk including
        chest pain type, cholesterol levels, ECG results, and exercise tolerance.
        """)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Cardiovascular Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input('Age', min_value=1,
                              max_value=120, value=50, key='heart_age')
        sex = st.selectbox('Sex', options=[
                           0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male', key='heart_sex')
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3],
                          format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'][x], key='heart_cp')

    with col2:
        trestbps = st.number_input(
            'Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120, key='heart_bp')
        chol = st.number_input(
            'Cholesterol (mg/dl)', min_value=100, max_value=600, value=200, key='heart_chol')
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1],
                           format_func=lambda x: 'No' if x == 0 else 'Yes', key='heart_fbs')

    with col3:
        restecg = st.selectbox('Rest ECG', options=[0, 1, 2],
                               format_func=lambda x: ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'][x], key='heart_ecg')
        thalach = st.number_input(
            'Max Heart Rate', min_value=60, max_value=220, value=150, key='heart_rate')
        exang = st.selectbox('Exercise Induced Angina', options=[0, 1],
                             format_func=lambda x: 'No' if x == 0 else 'Yes', key='heart_exang')

    with col4:
        oldpeak = st.number_input('ST Depression', min_value=0.0,
                                  max_value=10.0, value=1.0, step=0.1, key='heart_oldpeak')
        slope = st.selectbox('ST Slope', options=[0, 1, 2],
                             format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x], key='heart_slope')
        ca = st.number_input('Major Vessels (0-3)', min_value=0,
                             max_value=3, value=0, key='heart_ca')
        thal = st.selectbox('Thalassemia', options=[0, 1, 2],
                            format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect'][x], key='heart_thal')

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔍 Predict Heart Disease Risk', key='heart_predict'):
            if models['heart'] is not None:
                user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                              exang, oldpeak, slope, ca, thal]

                try:
                    prediction = models['heart'].predict([user_input])

                    if prediction[0] == 1:
                        st.markdown(
                            '<div class="prediction-result positive-result">⚠️ HIGH HEART DISEASE RISK</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="prediction-result negative-result">✅ LOW HEART DISEASE RISK</div>', unsafe_allow_html=True)

                    # Create user data dictionary for Gemini analysis
                    user_data = {
                        'age': age, 'sex': sex, 'chest_pain_type': cp, 'resting_bp': trestbps,
                        'cholesterol': chol, 'fasting_blood_sugar': fbs, 'rest_ecg': restecg,
                        'max_heart_rate': thalach, 'exercise_angina': exang, 'st_depression': oldpeak,
                        'st_slope': slope, 'major_vessels': ca, 'thalassemia': thal
                    }

                    # Get Gemini AI insights
                    if st.session_state.gemini_model:
                        with st.spinner("Generating personalized AI insights..."):
                            ai_insights = gemini_assistant.get_health_insights(
                                prediction[0], user_data, "heart disease")
                            st.markdown(
                                f'<div class="gemini-response"><h4>🤖 Gemini AI Health Insights</h4>{ai_insights}</div>', unsafe_allow_html=True)
                    else:
                        st.warning(
                            "🤖 Connect Gemini AI in the sidebar for personalized insights!")
                        with st.expander("What AI insights would include:", expanded=False):
                            st.markdown("""
                            - 📊 **Cardiovascular Risk Analysis** based on your metrics
                            - 🎯 **Key Risk Factors** from your heart health data
                            - 💡 **Heart-Healthy Lifestyle Tips** personalized for you
                            - 🏃 **Exercise Recommendations** for cardiovascular health
                            - 🥗 **Dietary Guidelines** for heart disease prevention
                            - 🏥 **When to Consult a Cardiologist**
                            """)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.error("Heart disease model not available!")

# Enhanced Kidney Disease Prediction
elif selected == '🫘 Kidney Disease':
    st.markdown('<div class="disease-card"><h2>🫘 Chronic Kidney Disease Prediction</h2><p>Advanced kidney health assessment with AI insights</p></div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About Kidney Disease Prediction", expanded=False):
        st.write("""
        This model analyzes key kidney function indicators including blood pressure, albumin levels,
        blood glucose, creatinine, and other vital parameters to assess kidney health.
        """)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Kidney Health Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        patient_id = st.number_input(
            'Patient ID', min_value=1, max_value=10000, value=1, key='ckd_id')
        age = st.number_input('Age', min_value=1,
                              max_value=120, value=50, key='ckd_age')
        bp = st.number_input('Blood Pressure (mm Hg)',
                             min_value=50, max_value=250, value=120, key='ckd_bp')

    with col2:
        al = st.number_input('Albumin (0-5)', min_value=0,
                             max_value=5, value=0, key='ckd_albumin')
        su = st.number_input('Sugar Level (0-5)', min_value=0,
                             max_value=5, value=0, key='ckd_sugar')
        bgr = st.number_input('Blood Glucose Random (mg/dl)',
                              min_value=50, max_value=500, value=120, key='ckd_glucose')

    with col3:
        bu = st.number_input('Blood Urea (mg/dl)', min_value=5,
                             max_value=300, value=40, key='ckd_urea')
        sc = st.number_input('Serum Creatinine (mg/dl)', min_value=0.1,
                             max_value=20.0, value=1.2, step=0.1, key='ckd_creatinine')
        sod = st.number_input('Sodium (mEq/L)', min_value=100,
                              max_value=200, value=140, key='ckd_sodium')

    with col4:
        hemo = st.number_input('Hemoglobin (g/dl)', min_value=3.0,
                               max_value=20.0, value=12.0, step=0.1, key='ckd_hemoglobin')
        pcv = st.number_input('Packed Cell Volume (%)',
                              min_value=10, max_value=60, value=40, key='ckd_pcv')

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔍 Predict Kidney Disease Risk', key='ckd_predict'):
            if models['ckd'] is not None:
                user_input = [patient_id, age, bp, al,
                              su, bgr, bu, sc, sod, hemo, pcv]

                try:
                    prediction = models['ckd'].predict([user_input])

                    if prediction[0] == 1:
                        st.markdown(
                            '<div class="prediction-result positive-result">⚠️ HIGH KIDNEY DISEASE RISK</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="prediction-result negative-result">✅ LOW KIDNEY DISEASE RISK</div>', unsafe_allow_html=True)

                    # Create user data dictionary for Gemini analysis
                    user_data = {
                        'age': age, 'blood_pressure': bp, 'albumin': al, 'sugar': su,
                        'blood_glucose': bgr, 'blood_urea': bu, 'creatinine': sc,
                        'sodium': sod, 'hemoglobin': hemo, 'pcv': pcv
                    }

                    # Get Gemini AI insights
                    if st.session_state.gemini_model:
                        with st.spinner("Generating personalized AI insights..."):
                            ai_insights = gemini_assistant.get_health_insights(
                                prediction[0], user_data, "chronic kidney disease")
                            st.markdown(
                                f'<div class="gemini-response"><h4>🤖 Gemini AI Health Insights</h4>{ai_insights}</div>', unsafe_allow_html=True)
                    else:
                        st.warning(
                            "🤖 Connect Gemini AI in the sidebar for personalized insights!")
                        with st.expander("What AI insights would include:", expanded=False):
                            st.markdown("""
                            - 📊 **Kidney Function Analysis** based on your lab values
                            - 🎯 **Key Risk Indicators** from your test results
                            - 💡 **Kidney-Friendly Diet** recommendations
                            - 💧 **Hydration Guidelines** for kidney health
                            - 🏥 **Follow-up Care** recommendations
                            - 🧪 **Important Tests** to monitor regularly
                            """)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.error("Kidney disease model not available!")

# Enhanced Cancer Prediction
elif selected == '🫁 Cancer':
    st.markdown('<div class="disease-card"><h2>🫁 Cancer Risk Prediction</h2><p>Comprehensive cancer risk assessment using lifestyle and environmental factors</p></div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About Cancer Risk Prediction", expanded=False):
        st.write("""
        This model analyzes lifestyle, environmental, and genetic factors to assess cancer risk.
        It considers factors like smoking, pollution exposure, genetic predisposition, and various symptoms.
        """)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Health and Lifestyle Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input('Age', min_value=1,
                              max_value=120, value=50, key='cancer_age')
        air_pollution = st.slider(
            'Air Pollution Level (1-8)', min_value=1, max_value=8, value=4, key='cancer_pollution')
        alcohol_use = st.slider(
            'Alcohol Use (1-8)', min_value=1, max_value=8, value=4, key='cancer_alcohol')
        dust_allergy = st.slider(
            'Dust Allergy (1-8)', min_value=1, max_value=8, value=4, key='cancer_dust')

    with col2:
        genetic_risk = st.slider(
            'Genetic Risk (1-7)', min_value=1, max_value=7, value=4, key='cancer_genetic')
        obesity = st.slider('Obesity Level (1-7)', min_value=1,
                            max_value=7, value=4, key='cancer_obesity')
        smoking = st.slider('Smoking (1-8)', min_value=1,
                            max_value=8, value=4, key='cancer_smoking')
        passive_smoker = st.slider(
            'Passive Smoking (1-8)', min_value=1, max_value=8, value=4, key='cancer_passive')

    with col3:
        chest_pain = st.slider(
            'Chest Pain (1-9)', min_value=1, max_value=9, value=4, key='cancer_chest')
        fatigue = st.slider('Fatigue (1-9)', min_value=1,
                            max_value=9, value=4, key='cancer_fatigue')
        weight_loss = st.slider(
            'Weight Loss (1-8)', min_value=1, max_value=8, value=4, key='cancer_weight')
        shortness_breath = st.slider(
            'Shortness of Breath (1-9)', min_value=1, max_value=9, value=4, key='cancer_breath')

    with col4:
        swallowing_difficulty = st.slider(
            'Swallowing Difficulty (1-8)', min_value=1, max_value=8, value=4, key='cancer_swallow')
        clubbing_nails = st.slider(
            'Clubbing of Finger Nails (1-9)', min_value=1, max_value=9, value=4, key='cancer_nails')
        frequent_cold = st.slider(
            'Frequent Cold (1-7)', min_value=1, max_value=7, value=4, key='cancer_cold')
        dry_cough = st.slider('Dry Cough (1-7)', min_value=1,
                              max_value=7, value=4, key='cancer_cough')
        snoring = st.slider('Snoring (1-7)', min_value=1,
                            max_value=7, value=4, key='cancer_snoring')

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔍 Predict Cancer Risk', key='cancer_predict'):
            if models['cancer'] is not None:
                user_input = [age, air_pollution, alcohol_use, dust_allergy, genetic_risk, obesity,
                              smoking, passive_smoker, chest_pain, fatigue, weight_loss, shortness_breath,
                              swallowing_difficulty, clubbing_nails, frequent_cold, dry_cough, snoring]

                try:
                    prediction = models['cancer'].predict([user_input])

                    if prediction[0] == 1:
                        st.markdown(
                            '<div class="prediction-result positive-result">⚠️ ELEVATED CANCER RISK DETECTED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="prediction-result negative-result">✅ LOW CANCER RISK</div>', unsafe_allow_html=True)

                    # Create user data dictionary for Gemini analysis
                    user_data = {
                        'age': age, 'air_pollution': air_pollution, 'alcohol_use': alcohol_use,
                        'dust_allergy': dust_allergy, 'genetic_risk': genetic_risk, 'obesity': obesity,
                        'smoking': smoking, 'passive_smoking': passive_smoker, 'chest_pain': chest_pain,
                        'fatigue': fatigue, 'weight_loss': weight_loss, 'shortness_breath': shortness_breath,
                        'swallowing_difficulty': swallowing_difficulty, 'clubbing_nails': clubbing_nails,
                        'frequent_cold': frequent_cold, 'dry_cough': dry_cough, 'snoring': snoring
                    }

                    # Get Gemini AI insights
                    if st.session_state.gemini_model:
                        with st.spinner("Generating personalized AI insights..."):
                            ai_insights = gemini_assistant.get_health_insights(
                                prediction[0], user_data, "cancer")
                            st.markdown(
                                f'<div class="gemini-response"><h4>🤖 Gemini AI Health Insights</h4>{ai_insights}</div>', unsafe_allow_html=True)
                    else:
                        st.warning(
                            "🤖 Connect Gemini AI in the sidebar for personalized insights!")
                        with st.expander("What AI insights would include:", expanded=False):
                            st.markdown("""
                            - 📊 **Risk Factor Analysis** of lifestyle and environmental factors
                            - 🎯 **Key Prevention Strategies** based on your profile
                            - 💡 **Lifestyle Modifications** to reduce cancer risk
                            - 🏥 **Screening Recommendations** for your age group
                            - 🚭 **Smoking Cessation** support if applicable
                            - 🥗 **Anti-cancer Diet** guidelines
                            """)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.error("Cancer prediction model not available!")

# Enhanced Parkinson's Prediction
elif selected == '🧠 Parkinson\'s':
    st.markdown('<div class="disease-card"><h2>🧠 Parkinson\'s Disease Prediction</h2><p>Advanced neurological assessment using voice and motor analysis</p></div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About Parkinson's Disease Prediction", expanded=False):
        st.write("""
        This model analyzes voice characteristics and motor patterns that are affected in Parkinson's disease.
        It examines vocal features like jitter, shimmer, and other speech-related biomarkers.
        """)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Voice and Motor Analysis Data")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=50.0,
                             max_value=300.0, value=150.0, step=0.1, key='park_fo')
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=100.0,
                              max_value=500.0, value=200.0, step=0.1, key='park_fhi')
        flo = st.number_input('MDVP:Flo(Hz)', min_value=50.0,
                              max_value=200.0, value=100.0, step=0.1, key='park_flo')
        jitter_percent = st.number_input(
            'MDVP:Jitter(%)', min_value=0.0, max_value=10.0, value=0.5, step=0.001, key='park_jitter_p')
        jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0,
                                     max_value=1.0, value=0.01, step=0.0001, key='park_jitter_a')

    with col2:
        rap = st.number_input('MDVP:RAP', min_value=0.0,
                              max_value=1.0, value=0.01, step=0.001, key='park_rap')
        ppq = st.number_input('MDVP:PPQ', min_value=0.0,
                              max_value=1.0, value=0.01, step=0.001, key='park_ppq')
        ddp = st.number_input('Jitter:DDP', min_value=0.0,
                              max_value=1.0, value=0.03, step=0.001, key='park_ddp')
        shimmer = st.number_input('MDVP:Shimmer', min_value=0.0,
                                  max_value=1.0, value=0.03, step=0.001, key='park_shimmer')
        shimmer_db = st.number_input(
            'MDVP:Shimmer(dB)', min_value=0.0, max_value=5.0, value=0.3, step=0.01, key='park_shimmer_db')

    with col3:
        apq3 = st.number_input('Shimmer:APQ3', min_value=0.0,
                               max_value=1.0, value=0.015, step=0.001, key='park_apq3')
        apq5 = st.number_input('Shimmer:APQ5', min_value=0.0,
                               max_value=1.0, value=0.02, step=0.001, key='park_apq5')
        apq = st.number_input('MDVP:APQ', min_value=0.0,
                              max_value=1.0, value=0.02, step=0.001, key='park_apq')
        dda = st.number_input('Shimmer:DDA', min_value=0.0,
                              max_value=1.0, value=0.045, step=0.001, key='park_dda')
        nhr = st.number_input(
            'NHR', min_value=0.0, max_value=1.0, value=0.02, step=0.001, key='park_nhr')

    with col4:
        hnr = st.number_input(
            'HNR', min_value=0.0, max_value=50.0, value=20.0, step=0.1, key='park_hnr')
        rpde = st.number_input(
            'RPDE', min_value=0.0, max_value=1.0, value=0.5, step=0.001, key='park_rpde')
        dfa = st.number_input(
            'DFA', min_value=0.0, max_value=1.0, value=0.7, step=0.001, key='park_dfa')
        spread1 = st.number_input(
            'spread1', min_value=-10.0, max_value=10.0, value=-5.0, step=0.1, key='park_spread1')
        spread2 = st.number_input(
            'spread2', min_value=0.0, max_value=1.0, value=0.2, step=0.001, key='park_spread2')

    with col5:
        d2 = st.number_input('D2', min_value=0.0, max_value=10.0,
                             value=2.0, step=0.01, key='park_d2')
        ppe = st.number_input(
            'PPE', min_value=0.0, max_value=1.0, value=0.2, step=0.001, key='park_ppe')

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔍 Predict Parkinson\'s Risk', key='parkinson_predict'):
            if models['parkinson'] is not None:
                user_input = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer,
                              shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]

                try:
                    prediction = models['parkinson'].predict([user_input])

                    if prediction[0] == 1:
                        st.markdown(
                            '<div class="prediction-result positive-result">⚠️ PARKINSON\'S RISK INDICATORS DETECTED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="prediction-result negative-result">✅ LOW PARKINSON\'S RISK</div>', unsafe_allow_html=True)

                    # Create user data dictionary for Gemini analysis
                    user_data = {
                        'vocal_fo': fo, 'vocal_fhi': fhi, 'vocal_flo': flo, 'jitter_percent': jitter_percent,
                        'jitter_abs': jitter_abs, 'rap': rap, 'ppq': ppq, 'shimmer': shimmer,
                        'hnr': hnr, 'rpde': rpde, 'dfa': dfa, 'ppe': ppe
                    }

                    # Get Gemini AI insights
                    if st.session_state.gemini_model:
                        with st.spinner("Generating personalized AI insights..."):
                            ai_insights = gemini_assistant.get_health_insights(
                                prediction[0], user_data, "Parkinson's disease")
                            st.markdown(
                                f'<div class="gemini-response"><h4>🤖 Gemini AI Health Insights</h4>{ai_insights}</div>', unsafe_allow_html=True)
                    else:
                        st.warning(
                            "🤖 Connect Gemini AI in the sidebar for personalized insights!")
                        with st.expander("What AI insights would include:", expanded=False):
                            st.markdown("""
                            - 📊 **Voice Pattern Analysis** and neurological indicators
                            - 🎯 **Early Detection Insights** from vocal biomarkers
                            - 💡 **Neurological Health Tips** and brain exercises
                            - 🏥 **Specialist Consultation** recommendations
                            - 🗣️ **Speech Therapy** guidance if needed
                            - 🧠 **Cognitive Health** maintenance strategies
                            """)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.error("Parkinson's disease model not available!")

# Footer with connection status
st.markdown("---")

if not st.session_state.gemini_model:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;'>
        <h3>🚀 Unlock AI-Powered Health Insights!</h3>
        <p>Connect your Gemini API key in the sidebar to access:</p>
        <div style='display: flex; justify-content: space-around; margin: 1rem 0; flex-wrap: wrap;'>
            <div>🤖 AI Chat Assistant</div>
            <div>🔍 Symptom Analysis</div>
            <div>📊 Personalized Insights</div>
            <div>💡 Health Recommendations</div>
        </div>
        <p><strong>Get your free API key:</strong> <a href='https://makersuite.google.com/app/apikey' target='_blank' style='color: #FFD700;'>Google AI Studio</a></p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.success(
        "🎉 Gemini AI is connected and ready to help with your health questions!")

st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>⚕️PredictaHealth AI</h4>
    <p>Machine Learning Predictions + AI Insights • For Educational Purposes Only</p>
    <p><strong>⚠️ Important:</strong> These predictions and AI insights are for informational purposes only. 
    Always consult with healthcare professionals for medical advice, diagnosis, and treatment.</p>
</div>
""", unsafe_allow_html=True)
