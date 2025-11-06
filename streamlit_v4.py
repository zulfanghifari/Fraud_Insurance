import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

# Load model and explainer
@st.cache_resource
def load_model():
    with open('final_model_xbg_tuned20251105_1056.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_explainer():
    with open('lime_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return explainer

try:
    model_pipeline = load_model()
    lime_explainer = load_explainer()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model or explainer: {str(e)}")
    model_loaded = False

# Title
st.title("🚗 Insurance Fraud Detection System")
st.markdown("---")

# Feature groupings
if model_loaded:
    # Create columns for better layout
    st.markdown("### Enter Policy and Claim Information")
    
    # Initialize input dictionary
    input_data = {}
    
    # Group 1: Temporal Information
    st.markdown("#### 📅 Temporal Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data['Month'] = st.selectbox(
            'Month of Accident',
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )
        input_data['DayOfWeek'] = st.selectbox(
            'Day of Week (Accident)',
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
    
    with col2:
        input_data['MonthClaimed'] = st.selectbox(
            'Month Claimed',
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '0']
        )
        input_data['DayOfWeekClaimed'] = st.selectbox(
            'Day of Week (Claimed)',
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', '0']
        )
    
    with col3:
        input_data['Year'] = st.number_input('Year', min_value=1994, max_value=1996, value=1995, step=1)
        input_data['WeekOfMonth'] = st.slider('Week of Month (Accident)', min_value=1, max_value=5, value=3)
        input_data['WeekOfMonthClaimed'] = st.slider('Week of Month (Claimed)', min_value=1, max_value=5, value=3)
    
    st.markdown("---")
    
    # Group 2: Policy Holder Information
    st.markdown("#### 👤 Policy Holder Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        input_data['Sex'] = st.selectbox('Sex', ['Male', 'Female'])
        input_data['MaritalStatus'] = st.selectbox('Marital Status', ['Single', 'Married', 'Widow', 'Divorced'])
    
    with col2:
        input_data['AgeOfPolicyHolder'] = st.selectbox(
            'Age of Policy Holder',
            ['16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 65', 'over 65']
        )
        input_data['Age'] = st.number_input('Age (Numerical)', min_value=0, max_value=80, value=40, step=1)
    
    with col3:
        input_data['DriverRating'] = st.slider('Driver Rating', min_value=1, max_value=4, value=2)
        input_data['PastNumberOfClaims'] = st.selectbox(
            'Past Number of Claims',
            ['none', '1', '2 to 4', 'more than 4']
        )
    
    with col4:
        input_data['NumberOfCars'] = st.selectbox(
            'Number of Cars',
            ['1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8']
        )
        input_data['AddressChange_Claim'] = st.selectbox(
            'Address Change Since Claim',
            ['no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years']
        )
    
    st.markdown("---")
    
    # Group 3: Vehicle Information
    st.markdown("#### 🚙 Vehicle Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        input_data['Make'] = st.selectbox(
            'Vehicle Make',
            ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 'Dodge', 
             'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus']
        )
        input_data['VehicleCategory'] = st.selectbox('Vehicle Category', ['Sport', 'Utility', 'Sedan'])
    
    with col2:
        input_data['VehiclePrice'] = st.selectbox(
            'Vehicle Price Range',
            ['less than 20000', '20000 to 29000', '30000 to 39000', '40000 to 59000', '60000 to 69000', 'more than 69000']
        )
        input_data['AgeOfVehicle'] = st.selectbox(
            'Age of Vehicle',
            ['new', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7']
        )
    
    with col3:
        input_data['Days_Policy_Accident'] = st.selectbox(
            'Days Between Policy & Accident',
            ['none', '1 to 7', '8 to 15', '15 to 30', 'more than 30']
        )
        input_data['Days_Policy_Claim'] = st.selectbox(
            'Days Between Policy & Claim',
            ['none', '8 to 15', '15 to 30', 'more than 30']
        )
    
    with col4:
        st.write("")  # Spacer
    
    st.markdown("---")
    
    # Group 4: Policy and Claim Details
    st.markdown("#### 📋 Policy and Claim Details")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        input_data['PolicyType'] = st.selectbox(
            'Policy Type',
            ['Sport - Liability', 'Sport - Collision', 'Sport - All Perils',
             'Sedan - Liability', 'Sedan - Collision', 'Sedan - All Perils',
             'Utility - Liability', 'Utility - Collision', 'Utility - All Perils']
        )
        input_data['BasePolicy'] = st.selectbox('Base Policy', ['Liability', 'Collision', 'All Perils'])
    
    with col2:
        input_data['PolicyNumber'] = st.number_input('Policy Number', min_value=1, max_value=15420, value=7710, step=1)
        input_data['Deductible'] = st.number_input('Deductible', min_value=300, max_value=700, value=400, step=50)
    
    with col3:
        input_data['RepNumber'] = st.number_input('Rep Number', min_value=1, max_value=16, value=8, step=1)
        input_data['NumberOfSuppliments'] = st.selectbox(
            'Number of Supplements',
            ['none', '1 to 2', '3 to 5', 'more than 5']
        )
    
    with col4:
        input_data['AgentType'] = st.selectbox('Agent Type', ['External', 'Internal'])
    
    st.markdown("---")
    
    # Group 5: Accident Details
    st.markdown("#### 🚨 Accident Details")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        input_data['AccidentArea'] = st.selectbox('Accident Area', ['Urban', 'Rural'])
        input_data['Fault'] = st.selectbox('Fault', ['Policy Holder', 'Third Party'])
    
    with col2:
        input_data['PoliceReportFiled'] = st.selectbox('Police Report Filed', ['Yes', 'No'])
        input_data['WitnessPresent'] = st.selectbox('Witness Present', ['Yes', 'No'])
    
    with col3:
        st.write("")  # Spacer
    
    with col4:
        st.write("")  # Spacer
    
    st.markdown("---")
    st.markdown("---")
    
    # Predict button centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔍 Predict Fraud", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Prediction and LIME explanation
    if predict_button:
        try:
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model_pipeline.predict(input_df)[0]
            prediction_proba = model_pipeline.predict_proba(input_df)[0]
            
            # Display prediction
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "FRAUD ⚠️" if prediction == 1 else "LEGITIMATE ✅")
            
            with col2:
                st.metric("Fraud Probability", f"{prediction_proba[1]:.2%}")
            
            with col3:
                st.metric("Legitimate Probability", f"{prediction_proba[0]:.2%}")
            
            # Progress bar for probability
            st.progress(float(prediction_proba[1]))
            
            if prediction == 1:
                st.error("⚠️ **HIGH RISK**: This claim shows signs of potential fraud. Further investigation recommended.")
            else:
                st.success("✅ **LOW RISK**: This claim appears to be legitimate.")
            
            st.markdown("---")
            
            # LIME Explanation
            st.markdown("### 🔬 LIME Explanation")
            st.markdown("*Understanding what factors influenced this prediction*")
            
            try:
                # Transform input using preprocessing step
                preprocessing_step = model_pipeline.named_steps['preprocessing']
                transformed_data = preprocessing_step.transform(input_df)
                
                # Get feature names after transformation
                if hasattr(transformed_data, 'columns'):
                    feature_names = transformed_data.columns.tolist()
                else:
                    # If not a DataFrame, get feature names from the preprocessing step
                    feature_names = preprocessing_step.get_feature_names_out().tolist()
                    transformed_data = pd.DataFrame(transformed_data, columns=feature_names)
                
                # Get the actual model (not the pipeline)
                actual_model = model_pipeline.named_steps['model']
                
                # Create LIME explanation
                exp = lime_explainer.explain_instance(
                    transformed_data.iloc[0].values,
                    actual_model.predict_proba,
                    num_features=10
                )
                
                # Create figure
                fig = exp.as_pyplot_figure()
                fig.set_size_inches(12, 6)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                st.markdown("""
                **How to read this chart:**
                - Features on the left show what influenced the prediction
                - Orange bars indicate factors pushing towards FRAUD
                - Blue bars indicate factors pushing towards LEGITIMATE
                - Longer bars = stronger influence on the prediction
                """)
                
            except Exception as e:
                st.error(f"Error generating LIME explanation: {str(e)}")
                st.info("The prediction is still valid. LIME explanation could not be generated.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all required fields are filled correctly.")

else:
    st.warning("⚠️ Model files not found. Please ensure the following files are in the same directory as this app:")
    st.code("""
    - final_model_xbg_tuned20251105_1056.pkl
    - lime_explainer.pkl
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Insurance Fraud Detection System | Powered by XGBoost & LIME</p>
</div>
""", unsafe_allow_html=True)