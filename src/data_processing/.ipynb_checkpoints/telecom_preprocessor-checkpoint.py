import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st # For st.warning/error messages in Streamlit context
from sklearn.preprocessing import MinMaxScaler # Needed for scaling value components

# --- REMOVED INCORRECT SELF-IMPORT:
# from data_processing.telecom_preprocessor import TelecomPreprocessor
# This line should NOT be in this file.
# ---

class TelecomPreprocessor:
    """
    A reusable class to apply the full preprocessing and feature engineering
    pipeline to raw telecom customer data. This ensures consistency between
    model training and new data prediction.
    """
    def __init__(self, scaler, feature_columns):
        """
        Initializes the preprocessor with a pre-fitted scaler and the list of expected features.
        
        Args:
            scaler: A pre-fitted StandardScaler object (loaded from your training phase).
            feature_columns: A list of strings, representing the final feature names
                             expected by the trained ML model (your 41 features).
        """
        if scaler is None:
            raise ValueError("Scaler must be provided and pre-fitted for preprocessing.")
        if feature_columns is None or not feature_columns:
            raise ValueError("Feature columns list must be provided and not empty.")
            
        self.scaler = scaler
        self.feature_columns = feature_columns
        
        # --- Define the RAW features your preprocessing expects ---
        self.expected_raw_features = [
            'Phone Number', 'Account Length', 'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge',
            'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge',
            'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls', 'Churn'
        ]
        
        # --- CRITICAL: ACTUAL TRAINING-DERIVED CONSTANTS FROM YOUR NOTEBOOK ---
        # These values MUST be derived from your ORIGINAL, large training dataset.
        # REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL CALCULATED VALUES FROM 01_telecom_data_preprocessing.ipynb
        
        # Medians for imputation (from your notebook's cleaning section)
        self.median_numeric_values_train = { 
            'Account Length': 100.0, 'VMail Message': 0.0, 'Day Mins': 179.775, 'Day Calls': 100.0,
            'Day Charge': 30.56, 'Eve Mins': 200.980, 'Eve Calls': 100.0, 'Eve Charge': 17.08,
            'Night Mins': 200.860, 'Night Calls': 100.0, 'Night Charge': 9.04, 'Intl Mins': 10.0,
            'Intl Calls': 4.0, 'Intl Charge': 2.70, 'CustServ Calls': 1.0
        }
        
        # Quantiles/Means for Behavioral Flags and Scores
        # These values MUST be fixed values derived from your original, large training dataset.
        self.mean_day_mins_train = 179.775 # Actual mean from training
        self.mean_eve_mins_train = 200.980 # Actual mean from training
        self.mean_night_mins_train = 200.860 # Actual mean from training
        self.quantile_75_day_mins_train = 230.1 # Actual 75th quantile from training
        self.quantile_75_eve_mins_train = 240.2 # Actual 75th quantile from training
        self.quantile_75_night_mins_train = 240.3 # Actual 75th quantile from training
        self.quantile_75_total_charges_train = 60.0 # Actual 75th percentile from training
        self.quantile_80_total_charges_train = 65.0 # Actual 80th percentile from training
        self.median_total_minutes_train = 500.0 # Actual median from training data
        self.quantile_25_total_minutes_train = 300.0 # Actual 25th quantile from training data
        self.mean_total_minutes_train_for_risk = 500.0 # Actual mean from training (used in Risk Score calc, distinct name)
        self.quantile_75_account_length_train = 150.0 # Actual 75th quantile from training
        
        # Max values for scaling scores
        self.max_custserv_calls_train = 9.0 # Actual max from training data (used for scaling Satisfaction_Score)
        self.max_total_charges_train = 100.0 # Actual max from training data (used for Customer_Value_Score)
        self.max_total_minutes_train = 1500.0 # Actual max from training data (used for Customer_Value_Score)
        self.max_account_length_train = 250.0 # Actual max from training data (used for Customer_Value_Score)
        
        # Pre-calculate max_churn_risk_score_train_derived for consistency
        # This needs to be a fixed value from training, not derived from current df
        self.max_churn_risk_score_train_derived = (self.max_custserv_calls_train * 0.4) + \
                                                  (0.3 if (self.mean_total_minutes_train_for_risk * 0.5) < (self.mean_total_minutes_train + 1e-6) else 0) + \
                                                  (0.3 if 100 < self.max_account_length_train else 0) # Adjust based on your actual derivation

    def transform(self, df_raw: pd.DataFrame) -> (np.ndarray, pd.DataFrame):
        df_processed = df_raw.copy()
        
        # --- 1. Data Cleaning Process (From your notebook snippet) ---
        
        # Ensure 'Phone Number' is handled if present (not a feature for model prediction)
        if 'Phone Number' not in df_processed.columns:
            df_processed['Phone Number'] = [f'UPLOADED_{i}' for i in range(len(df_processed))]
        
        # Remove duplicates (important for consistency, though less likely in small uploaded batches)
        df_processed = df_processed.drop_duplicates(subset=[col for col in df_processed.columns if col != 'Phone Number']) 

        # Handle missing values (using medians from training data if possible)
        numeric_columns = [col for col in self.expected_raw_features if col in df_processed.columns and col not in ['Phone Number', 'Churn']]
        for col in numeric_columns:
            if df_processed[col].isnull().sum() > 0:
                median_val = self.median_numeric_values_train.get(col, df_processed[col].median()) 
                df_processed[col].fillna(median_val, inplace=True)
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0) # Ensure numeric type and fill any new NaNs

        # Convert Churn to binary
        if 'Churn' in df_processed.columns:
            df_processed['Churn_Binary'] = df_processed['Churn'].map({
                'FALSE': 0, 'True': 1, False: 0, True: 1
            })
            df_processed['Churn_Binary'] = df_processed['Churn_Binary'].fillna(0).astype(int)
        else:
            df_processed['Churn_Binary'] = 0 # Default if 'Churn' column is missing

        # --- 2. Feature Engineering (Your logic, explicitly ordered for dependencies) ---
        
        # Basic Aggregations (needed for many subsequent features)
        df_processed['Total_Minutes'] = (df_processed['Day Mins'] + df_processed['Eve Mins'] + 
                               df_processed['Night Mins'] + df_processed['Intl Mins'])
        df_processed['Total_Calls'] = (df_processed['Day Calls'] + df_processed['Eve Calls'] + 
                             df_processed['Night Calls'] + df_processed['Intl Calls'])
        df_processed['Total_Charges'] = (df_processed['Day Charge'] + df_processed['Eve Charge'] + 
                               df_processed['Night Charge'] + df_processed['Intl Charge'])

        # Lifecycle Features
        df_processed['Account_Length_Months'] = df_processed['Account Length'] / 30.0

        # Usage Intensity Features (handle division by zero with 1e-6)
        df_processed['Avg_Call_Duration'] = df_processed['Total_Minutes'] / (df_processed['Total_Calls'] + 1e-6)
        df_processed['Day_Call_Duration'] = df_processed['Day Mins'] / (df_processed['Day Calls'] + 1e-6)
        df_processed['Eve_Call_Duration'] = df_processed['Eve Mins'] / (df_processed['Eve Calls'] + 1e-6)
        df_processed['Night_Call_Duration'] = df_processed['Night Mins'] / (df_processed['Night Calls'] + 1e-6)
        df_processed['Intl_Call_Duration'] = df_processed['Intl Mins'] / (df_processed['Intl Calls'] + 1e-6) 
        
        # Usage Pattern Ratios (handle division by zero with 1e-6)
        df_processed['Day_Usage_Ratio'] = df_processed['Day Mins'] / (df_processed['Total_Minutes'] + 1e-6)
        df_processed['Eve_Usage_Ratio'] = df_processed['Eve Mins'] / (df_processed['Total_Minutes'] + 1e-6)
        df_processed['Night_Usage_Ratio'] = df_processed['Night Mins'] / (df_processed['Total_Minutes'] + 1e-6)
        df_processed['Intl_Usage_Ratio'] = df_processed['Intl Mins'] / (df_processed['Total_Minutes'] + 1e-6)
        
        # Revenue-Based Features (handle division by zero with 1e-6)
        df_processed['Revenue_Per_Minute'] = df_processed['Total_Charges'] / (df_processed['Total_Minutes'] + 1e-6)
        df_processed['Day_Revenue_Rate'] = df_processed['Day Charge'] / (df_processed['Day Mins'] + 1e-6)
        df_processed['Eve_Revenue_Rate'] = df_processed['Eve Charge'] / (df_processed['Eve Mins'] + 1e-6)
        df_processed['Night_Revenue_Rate'] = df_processed['Night Charge'] / (df_processed['Night Mins'] + 1e-6)
        
        # More Lifecycle Features (handle division by zero with 1e-6)
        df_processed['Usage_Per_Day'] = df_processed['Total_Minutes'] / (df_processed['Account Length'] + 1e-6)
        df_processed['Revenue_Per_Day'] = df_processed['Total_Charges'] / (df_processed['Account Length'] + 1e-6)
        df_processed['Service_Calls_Per_Month'] = df_processed['CustServ Calls'] / (df_processed['Account_Length_Months'] + 1e-6)

        # Behavioral Indicators (Binary Flags) - Use fixed thresholds from training data
        df_processed['Is_Heavy_Day_User'] = (df_processed['Day Mins'] > self.quantile_75_day_mins_train).astype(int)
        df_processed['Is_Heavy_Eve_User'] = (df_processed['Eve Mins'] > self.quantile_75_eve_mins_train).astype(int)
        df_processed['Is_Heavy_Night_User'] = (df_processed['Night Mins'] > self.quantile_75_night_mins_train).astype(int)
        df_processed['Is_Intl_User'] = (df_processed['Intl Mins'] > 0).astype(int)
        df_processed['Is_High_Service_User'] = (df_processed['CustServ Calls'] > 2).astype(int)
        df_processed['Has_Voicemail'] = (df_processed['VMail Message'] > 0).astype(int)
        df_processed['Is_High_Value_Customer'] = (df_processed['Total_Charges'] > self.quantile_80_total_charges_train).astype(int)

        # Advanced ML Features (Order matters for dependencies)
        df_processed['Day_Intensity'] = df_processed['Day Mins'] / (df_processed['Day Calls'] + 1e-6)
        df_processed['Eve_Intensity'] = df_processed['Eve Mins'] / (df_processed['Eve Calls'] + 1e-6)
        df_processed['Night_Intensity'] = df_processed['Night Mins'] / (df_processed['Night Calls'] + 1e-6)
        df_processed['Intl_Intensity'] = df_processed['Intl Mins'] / (df_processed['Intl Calls'] + 1e-6)
        
        # --- CRITICAL: Create ALL Score Columns in correct order BEFORE clipping loop ---
        
        # Satisfaction Score (depends on CustServ Calls, Churn_Binary, Account Length, Total_Minutes)
        df_processed['Satisfaction_Score'] = (
            (df_processed['CustServ Calls'] == 0) * 0.4 +           # No complaints = very satisfied
            (df_processed['Churn_Binary'] == 0) * 0.3 +             # Not churning = satisfied
            (df_processed['Account Length'] > self.quantile_75_account_length_train) * 0.2 + # Long tenure = loyal
            (df_processed['Total_Minutes'] > self.median_total_minutes_train) * 0.1           # Active usage = engaged
        )
        
        # Customer Value Score (depends on Total_Charges, Account Length, Total_Minutes)
        df_processed['Customer_Value_Score'] = (
            df_processed['Total_Charges'] / (self.max_total_charges_train + 1e-6) * 0.4 + 
            df_processed['Account Length'] / (self.max_account_length_train + 1e-6) * 0.3 + 
            df_processed['Total_Minutes'] / (self.max_total_minutes_train + 1e-6) * 0.3
        )
        
        # Risk Score (depends on Churn_Binary, CustServ Calls, Total_Minutes, Account Length)
        df_processed['Churn_Risk_Score'] = (
            (df_processed['Churn_Binary'] == 1) * 0.5 +             # Currently churning
            (df_processed['CustServ Calls'] >= 3) * 0.3 +           # High service issues (fixed threshold from notebook)
            (df_processed['Total_Minutes'] < self.quantile_25_total_minutes_train) * 0.2  # Low usage (fixed quantile from training)
        )

        # Engagement Score (depends on Has_Voicemail, Is_Intl_User, CustServ Calls)
        df_processed['Engagement_Score'] = (
            df_processed['Has_Voicemail'] * 0.2 +
            df_processed['Is_Intl_User'] * 0.3 +
            (df_processed['CustServ Calls'] == 0) * 0.5 # Fixed logic from notebook
        )
        
        # Create Upsell_Propensity (depends on Is_High_Value_Customer, Is_Heavy_Day_User, Is_Heavy_Eve_User, Is_Intl_User, Churn_Risk_Score)
        # Needs to use fixed max_churn_risk_score_train_derived from __init__ for consistency
        max_churn_risk_score_train_derived = (self.max_custserv_calls_train * 0.4) + \
                                             (0.3 if (self.mean_total_minutes_train_for_risk * 0.5) < (self.mean_total_minutes_train_for_risk + 1e-6) else 0) + \
                                             (0.3 if 100 < self.max_account_length_train else 0) # Simplified for mock. Needs to be fixed from training.
        
        df_processed['Upsell_Propensity'] = (
            df_processed['Is_High_Value_Customer'] * 0.3 +
            df_processed['Is_Heavy_Day_User'] * 0.2 +
            df_processed['Is_Heavy_Eve_User'] * 0.2 +
            df_processed['Is_Intl_User'] * 0.2 +
            (1 - df_processed['Churn_Risk_Score'] / (max_churn_risk_score_train_derived + 1e-6)) * 0.1
        )
        
        # Recommended products (not used by ML model directly, but good for display)
        conditions = [
            (df_processed['Is_Heavy_Day_User'] == 1) & (df_processed['Day_Usage_Ratio'] > 0.4),
            (df_processed['Is_Heavy_Eve_User'] == 1) & (df_processed['Eve_Usage_Ratio'] > 0.4),
            (df_processed['Is_Heavy_Night_User'] == 1) & (df_processed['Night_Usage_Ratio'] > 0.4),
            (df_processed['Is_Intl_User'] == 1) & (df_processed['Intl_Usage_Ratio'] > 0.1),
            (df_processed['Is_High_Service_User'] == 1),
            (df_processed['Has_Voicemail'] == 1) & (df_processed['VMail Message'] > 20)
        ]
        
        choices = [
            'Unlimited Day Plan', 'Evening Special Package', 'Night Owl Plan',
            'International Package', 'Premium Support Plan', 'Enhanced Voicemail Package'
        ]
        df_processed['Recommended_Product'] = np.select(conditions, choices, default='Standard Upgrade')
        
        # --- Final Clipping and NaN Handling for ALL Scores ---
        # This loop now runs *after* all scores are guaranteed to be created.
        for score_col in ['Satisfaction_Score', 'Customer_Value_Score', 'Risk_Score', 'Engagement_Score', 'Upsell_Propensity', 'Churn_Risk_Score']:
            # Added check for column existence before clipping/filling
            if score_col in df_processed.columns: 
                df_processed[score_col] = df_processed[score_col].clip(0, 1).fillna(0)
            else:
                # This should ideally not happen if feature engineering is complete
                # It indicates a fundamental mismatch in feature creation.
                st.warning(f"Score column '{score_col}' not found before final clipping. Setting to 0. This may indicate an issue in feature engineering.")
                df_processed[score_col] = 0.0


        # --- 3. Final Data Cleaning and Feature Selection ---
        # Fill any remaining NaNs (e.g., from divisions) with 0 and handle inf/-inf
        df_processed = df_processed.fillna(0).replace([np.inf, -np.inf], 0)

        # Select only the features the model expects (self.feature_columns)
        # It's CRITICAL that all features in self.feature_columns exist in df_processed
        missing_cols_for_model = [col for col in self.feature_columns if col not in df_processed.columns]
        if missing_cols_for_model:
            st.error(f"Preprocessing ERROR: Features {missing_cols_for_model} expected by the model are MISSING after feature engineering. Cannot proceed.")
            st.stop()
            
        # Ensure the order of columns matches the training data
        final_features_df = df_processed[self.feature_columns]
        
        # --- 4. Scaling ---
        scaled_features = self.scaler.transform(final_features_df)
        
        return scaled_features, df_processed # Return scaled features and the processed DF for display

