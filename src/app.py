import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import time

# Set page config
st.set_page_config(
    page_title="DreamHome Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        border: none;
        color: white;
    }
    h1 {
        color: #1E1E1E;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Paths
MODEL_PATH = 'models/model.pkl'
FEATURES_PATH = 'models/feature_names.json'

# Load artifacts
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = '../models/model.pkl'
    FEATURES_PATH = '../models/feature_names.json'

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
    return model, feature_names

try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error("⚠️ Model artifacts not found. Please run the training script first.")
    st.stop()

# --- Sidebar Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/25/25694.png", width=50)
    st.title("Property Details")
    st.markdown("Configure the house parameters below.")
    
    with st.expander("📍 Location & Quality", expanded=True):
        neighborhood = st.selectbox(
            "Neighborhood", 
            ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'NridgHt', 'Gilbert', 'StoneBr'],
            help="Select the neighborhood area of the property."
        )
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6, help="Rates the overall material and finish of the house")
        
    with st.expander("🏠 Size & Layout", expanded=True):
        gr_liv_area = st.number_input("Living Area (sqft)", 500, 10000, 1800, step=50)
        total_bsmt_sf = st.number_input("Basement Area (sqft)", 0, 5000, 800, step=50)
        garage_cars = st.slider("Garage Capacity (Cars)", 0, 5, 2)
        
    with st.expander("🏗️ Condition & Age"):
        year_built = st.number_input("Year Built", 1850, 2025, 2005)
        year_sold = st.number_input("Year Sold", 2000, 2025, 2010)
        bsmt_qual = st.selectbox("Basement Height", ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], index=1)
        kitchen_qual = st.selectbox("Kitchen Qual", ['Ex', 'Gd', 'TA', 'Fa'], index=1)
        
    predict_btn = st.button("✨ Estimate Price")

# --- Main Area ---
st.title("🏡 DreamHome Value Estimator")
st.markdown("### AI-Powered Real Estate Valuation")
st.markdown("Using advanced machine learning (XGBoost) to predict house prices based on historical data.")

col_hero1, col_hero2 = st.columns([2, 1])
with col_hero1:
    st.info("👈 **Start by adjusting the parameters in the sidebar!**")

if predict_btn:
    with st.spinner("Analyzing market trends..."):
        time.sleep(0.5) # UX delay
        
        # Data Prep
        data = {
            'OverallQual': overall_qual,
            'YearBuilt': year_built,
            'YrSold': year_sold,
            'GrLivArea': gr_liv_area,
            'TotalBsmtSF': total_bsmt_sf,
            'GarageCars': garage_cars,
            'Neighborhood': neighborhood,
            'BsmtQual': bsmt_qual,
            'KitchenQual': kitchen_qual,
            '1stFlrSF': gr_liv_area * 0.6,
            '2ndFlrSF': gr_liv_area * 0.4,
            'OpenPorchSF': 0, '3SsnPorch': 0, 'EnclosedPorch': 0, 
            'ScreenPorch': 0, 'WoodDeckSF': 0,
            'GarageYrBlt': year_built
        }
        
        df_in = pd.DataFrame([data])
        
        # Feature Engineering
        df_in['TotalSF'] = df_in['TotalBsmtSF'] + df_in['1stFlrSF'] + df_in['2ndFlrSF']
        df_in['TotalPorchSF'] = 0
        df_in['HouseAge'] = df_in['YrSold'] - df_in['YearBuilt']
        df_in['GarageAge'] = df_in['YrSold'] - df_in['GarageYrBlt']
        
        # Encoding
        df_in = pd.get_dummies(df_in)
        
        # Alignment
        df_final = pd.DataFrame(0, index=[0], columns=feature_names)
        for col in df_in.columns:
            if col in df_final.columns:
                df_final.at[0, col] = df_in.iloc[0][col]
        
        df_final = df_final.fillna(0)
        
        # Prediction
        try:
            log_pred = model.predict(df_final)[0]
            price = np.expm1(log_pred)
            
            # Display Result
            st.markdown("---")
            st.success("✅ Valuation Complete!")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin-bottom:0; color:#555;">Estimated Value</h3>
                    <h1 style="color:#2E86C1; margin-top:0;">${price:,.0f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            # Details
            st.markdown("#### 📊 Property Highlights")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("Living Area", f"{gr_liv_area:,} sqft")
            metrics_col2.metric("House Age", f"{df_in['HouseAge'][0]} years")
            metrics_col3.metric("Quality Score", f"{overall_qual}/10")
            
            # Feature Importance (Proxy)
            st.caption(f"Prediction based on {len(feature_names)} market factors.")
            st.balloons()
            
        except Exception as e:
            st.error(f"Something went wrong: {e}")
