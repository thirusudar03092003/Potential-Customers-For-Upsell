import pandas as pd
import joblib
import json
import streamlit as st
import os
import sys
import numpy as np # Ensure numpy is imported for random functions

# --- IMPORTANT: Corrected Path Definitions ---
# These paths are defined relative to the 'data_loader.py' file's location:
# src/dashboard/utils/data_loader.py

# Path to 'telecom_processed.csv':
# From utils/ -> ../../../data/processed/telecom_processed.csv
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'processed', 'telecom_processed.csv')

# Path to 'feature_columns.pkl':
# From utils/ -> ../../../notebooks/models/feature_columns.pkl
FEATURE_COLUMNS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'notebooks', 'models', 'feature_columns.pkl')

# Path to 'xgboost_smart_segmentation.pkl':
# From utils/ -> ../../../notebooks/models/optimized/xgboost_smart_segmentation.pkl
OPTIMIZED_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'notebooks', 'models', 'optimized', 'xgboost_smart_segmentation.pkl')

# Path to 'scaler.pkl':
# From utils/ -> ../../../notebooks/models/scaler.pkl
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'notebooks', 'models', 'scaler.pkl')
# --- END Corrected Path Definitions ---


@st.cache_data
def load_segmentation_data():
    """Load customer segmentation data"""
    try:
        # Load main dataset
        customers_df = pd.read_csv(DATA_PATH)
        
        # --- Placeholder/Fallback for UI Expected Columns ---
        # These ensure the UI doesn't crash if columns are unexpectedly missing
        if 'Customer_Category' not in customers_df.columns:
            st.warning("Column 'Customer_Category' not found, creating dummy categories for UI.")
            customers_df['Customer_Category'] = np.random.choice(
                ['STANDARD_UPSELL', 'DO_NOT_DISTURB', 'PRIORITY_UPSELL_RETENTION', 'FIX_FIRST_THEN_UPSELL', 'GENTLE_UPSELL', 'MINIMAL_CONTACT'], # More categories for realism
                size=len(customers_df)
            )
        if 'Priority_Level' not in customers_df.columns:
            st.warning("Column 'Priority_Level' not found, creating dummy levels for UI.")
            customers_df['Priority_Level'] = np.random.choice(
                ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'], # All priority levels
                size=len(customers_df)
            )
        if 'Priority_Score' not in customers_df.columns:
            st.warning("Column 'Priority_Score' not found, creating dummy scores for UI.")
            customers_df['Priority_Score'] = np.random.randint(1, 100, size=len(customers_df))
        
        # Add placeholder columns if they are expected by the UI but might be missing in raw data
        # Ensure these match the columns used in your UI (e.g., in show_customer_segments, show_priority_customers)
        for col in ['Total_Charges', 'Satisfaction_Score', 'Risk_Score', 'Account Length', 'Avg_Call_Duration', 'CustServ Calls', 'Customer_Value_Score']:
            if col not in customers_df.columns:
                st.warning(f"Column '{col}' not found, creating dummy values for UI.")
                customers_df[col] = np.random.rand(len(customers_df)) * 100 # Example dummy data
        
        # Ensure 'Has_Voicemail' is present for the prediction tool if it's a feature
        if 'Has_Voicemail' not in customers_df.columns:
            customers_df['Has_Voicemail'] = np.random.randint(0, 2, size=len(customers_df))


        return customers_df
        
    except Exception as e:
        st.error(f"Error loading data from '{DATA_PATH}': {e}")
        return pd.DataFrame()

@st.cache_resource
def load_prediction_model():
    """Load the optimized prediction model and associated artifacts"""
    try:
        model = joblib.load(OPTIMIZED_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        
        # Fallback for category_mapping if not loaded initially from a file
        category_mapping = {
            0: 'DO_NOT_DISTURB', 1: 'FIX_FIRST_THEN_UPSELL', 2: 'GENTLE_UPSELL',
            3: 'MINIMAL_CONTACT', 4: 'PRIORITY_UPSELL_RETENTION', 5: 'STANDARD_UPSELL'
        }
        
        return model, scaler, feature_columns, category_mapping
        
    except Exception as e:
        st.error(f"Error loading model artifacts from paths like '{OPTIMIZED_MODEL_PATH}' or '{FEATURE_COLUMNS_PATH}': {e}")
        return None, None, None, None

# The calculate_segment_metrics function is now in business_logic.py
# (It was moved there in a previous step for better organization)
