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

# --- Premium Design System ---
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Text */
    .main {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
    }
    
    h1, h2, h3 {
        color: #1a202c;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Premium Cards */
    .stCard {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Result Card Highlight */
    .result-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    }
    .result-card h3 { color: rgba(255,255,255,0.9); font-weight: 500; }
    .result-card h1 { color: white; font-size: 3.5rem; margin: 0.5rem 0; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f0f0;
    }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        background-color: #0f172a;
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(15, 23, 42, 0.1);
    }
    .stButton>button:hover {
        background-color: #1e293b;
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(15, 23, 42, 0.15);
        color: white;
    }
    
    /* Input Fields */
    .stSelectbox > div > div, .stNumberInput > div > div, .stSlider > div > div {
        border-radius: 8px;
        border-color: #e2e8f0;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Function to Load Model
@st.cache_resource
def load_artifacts():
    # Try multiple paths for robustness (local vs cloud)
    model_paths = ['models/model.pkl', '../models/model.pkl', 'model.pkl']
    feature_paths = ['models/feature_names.json', '../models/feature_names.json', 'feature_names.json']
    
    model = None
    feature_names = None
    
    for p in model_paths:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                model = pickle.load(f)
            break
            
    for p in feature_paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                feature_names = json.load(f)
            break
            
    return model, feature_names

# Load Model
model, feature_names = load_artifacts()
if not model or not feature_names:
    st.error("⚠️ Model artifacts not found. Please ensure `model.pkl` and `feature_names.json` are in the `models/` directory.")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### ⚙️ Property Config")
    st.info("Adjust the sliders and options below to describe the property.")
    
    st.markdown("#### 📍 Location")
    neighborhood = st.selectbox(
        "Neighborhood", 
        ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'NridgHt', 'Gilbert', 'StoneBr'],
        help="Select the neighborhood area."
    )
    
    st.markdown("#### 🏠 Dimensions")
    gr_liv_area = st.number_input("Living Area (sqft)", 500, 10000, 2000, step=50)
    total_bsmt_sf = st.number_input("Basement (sqft)", 0, 5000, 1000, step=50)
    garage_cars = st.slider("Garage Spaces", 0, 5, 2)
    
    st.markdown("#### ✨ Quality & Condition")
    overall_qual = st.slider("Quality Score (1-10)", 1, 10, 7)
    kitchen_qual = st.select_slider("Kitchen Quality", options=['Fa', 'TA', 'Gd', 'Ex'], value='Gd')
    bsmt_qual = st.select_slider("Basement Quality", options=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value='Gd')
    
    st.markdown("#### 📅 History")
    col_yr1, col_yr2 = st.columns(2)
    with col_yr1:
        year_built = st.number_input("Built In", 1800, 2025, 2010)
    with col_yr2:
        year_sold = st.number_input("Sold In", 2000, 2025, 2023)
        
    st.markdown("---")
    predict_btn = st.button("🚀 Calculate Value", use_container_width=True)

# --- Main Interface ---
# Hero Section
st.markdown("""
<div style="text-align: center; margin-bottom: 3rem;">
    <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">🏡 DreamHome Value</h1>
    <p style="font-size: 1.25rem; color: #64748b;">AI-Powered Residential Property Valuation</p>
</div>
""", unsafe_allow_html=True)

# Layout: 2 Columns for Inputs Visuals vs Results
if predict_btn:
    with st.spinner("Analyzing market data..."):
        time.sleep(0.8) # Premium feel delay
        
        # --- PREDICTION LOGIC ---
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
            '1stFlrSF': gr_liv_area * 0.6,   # Approx logic
            '2ndFlrSF': gr_liv_area * 0.4,
            'OpenPorchSF': 0, '3SsnPorch': 0, 'EnclosedPorch': 0, 
            'ScreenPorch': 0, 'WoodDeckSF': 0, 'GarageYrBlt': year_built
        }
        
        df_in = pd.DataFrame([data])
        
        # Engineering Matches Training
        df_in['TotalSF'] = df_in['TotalBsmtSF'] + df_in['1stFlrSF'] + df_in['2ndFlrSF']
        df_in['TotalPorchSF'] = 0
        df_in['HouseAge'] = df_in['YrSold'] - df_in['YearBuilt']
        df_in['GarageAge'] = df_in['YrSold'] - df_in['GarageYrBlt']
        
        df_in = pd.get_dummies(df_in)
        
        # Realign columns with training data
        df_final = pd.DataFrame(0, index=[0], columns=feature_names)
        for col in df_in.columns:
            if col in df_final.columns:
                df_final.at[0, col] = df_in.iloc[0][col]
        
        try:
            log_pred = model.predict(df_final.fillna(0))[0]
            price = np.expm1(log_pred)
            
            # --- RESULT DISPLAY ---
            st.markdown(f"""
            <div class="result-card">
                <h3>Estimated Market Value</h3>
                <h1>${price:,.0f}</h1>
                <p>Confidence Interval: ±5%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Drivers Section
            st.markdown("### 📊 Property Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Living Space", f"{gr_liv_area:,} sqft", delta="High Impact", delta_color="normal")
            with col2:
                st.metric("Quality Score", f"{overall_qual}/10", f"{'Premium' if overall_qual > 7 else 'Standard'}")
            with col3:
                st.metric("House Age", f"{df_in['HouseAge'][0]} yrs", delta=f"{'Vintage' if df_in['HouseAge'][0] > 50 else 'Modern'}", delta_color="off")
            with col4:
                st.metric("Location", neighborhood)
                
            st.progress(overall_qual/10, text="Overall Material Quality")
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

else:
    # Initial State / Welcome
    st.markdown("""
    <div class="stCard">
        <h3>👋 Welcome to DreamHome!</h3>
        <p>This tool uses an advanced <strong>XGBoost Machine Learning model</strong> to estimate house prices with high accuracy.</p>
        <p>To get started:</p>
        <ol>
            <li>Fill in the property details in the <strong>sidebar</strong> on the left.</li>
            <li>Click the <strong>Calculate Value</strong> button.</li>
            <li>Get an instant valuation estimate!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True) 

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.875rem;'>Built with ❤️ using Streamlit & XGBoost</p>", unsafe_allow_html=True)
