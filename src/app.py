import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # Needed for isinstance check
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
st.set_page_config(
    page_title="AI Upsell Prediction Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced UI/UX ---
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #0d1117; /* Dark background */
        color: #c9d1d9; /* Light text */
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Header styling */
    .stApp header {
        background-color: #161b22;
        border-bottom: 1px solid #30363d;
        padding: 1rem;
    }
    h1 {
        color: #58a6ff; /* Blue for emphasis */
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
        border-bottom: 2px solid #58a6ff;
        padding-bottom: 0.2em;
    }
    h2 {
        color: #8b949e;
        font-size: 1.8em;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }
    h3 {
        color: #c9d1d9;
        font-size: 1.4em;
        margin-top: 1em;
        margin-bottom: 0.6em;
    }
    
    /* Sidebar styling */
    .st-emotion-cache-vk329v { /* Target sidebar background */
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .st-emotion-cache-1tmxueq { /* Target sidebar header */
        color: #58a6ff;
    }

    /* Metric cards - Adjusted for better text display */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stMetric"] label {
        color: #8b949e;
        font-size: 1em;
        overflow: hidden; /* Ensure text doesn't overflow */
        white-space: nowrap; /* Prevent wrapping */
        text-overflow: ellipsis; /* Add ellipsis if cut off */
    }
    [data-testid="stMetric"] div[data-testid="stMarkdownContainer"] p {
        color: #58a6ff;
        font-size: 1.8em;
        font-weight: bold;
        overflow: hidden; /* Ensure text doesn't overflow */
        white-space: nowrap; /* Prevent wrapping */
        text-overflow: ellipsis; /* Add ellipsis if cut off */
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #238636; /* Green for action */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 1em;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Info/Success/Error messages */
    .stAlert {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stAlert.info { background-color: #0d47a1; color: #bbdefb; } /* Darker blue info */
    .stAlert.success { background-color: #1b5e20; color: #c8e6c9; } /* Darker green success */
    .stAlert.error { background-color: #b71c1c; color: #ffcdd2; } /* Darker red error */

    /* Spinner */
    .stSpinner > div > div {
        border-top-color: #58a6ff;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #30363d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        background-color: #161b22;
    }
    [data-testid="stFileUploader"] label {
        color: #8b949e;
    }
    [data-testid="stFileUploader"] button {
        background-color: #58a6ff;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #79c0ff;
    }

    /* Plot titles */
    .stPlotlyChart, .stMatplotlib {
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model Components ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model_components():
    try:
        model_path = 'models/best_model.pkl'
        scaler_path = 'models/scaler.pkl'
        features_path = 'models/feature_columns.pkl'

        best_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(features_path)
        
        return best_model, scaler, feature_columns
    except FileNotFoundError:
        st.error("❌ Error: Model components not found. Please ensure 'best_model.pkl', 'scaler.pkl', and 'feature_columns.pkl' are in the 'models/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading model components: {e}")
        st.stop()

best_model, scaler, feature_columns = load_model_components()

# --- Data Preprocessing Function (Reusing Logic from Phase 2) ---
def preprocess_data_for_prediction(df_raw_input):
    df_clean = df_raw_input.copy()

    # Handle duplicates (as found in EDA)
    df_clean = df_clean.drop_duplicates()

    # Handle missing values in numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # Handle missing values in 'Churn' column specifically
    if 'Churn' in df_clean.columns and df_clean['Churn'].isnull().any():
        df_clean['Churn'].fillna(False, inplace=True)
    
    # --- Feature Engineering (Reusing Logic from Phase 2) ---
    df_features = df_clean.copy()

    # 1. Usage aggregation features
    df_features['Total_Minutes'] = df_features['Day Mins'] + df_features['Eve Mins'] + df_features['Night Mins']
    df_features['Total_Calls'] = df_features['Day Calls'] + df_features['Eve Calls'] + df_features['Night Calls']
    df_features['Total_Charges'] = df_features['Day Charge'] + df_features['Eve Charge'] + df_features['Night Charge'] + df_features['Intl Charge']
    
    # 2. Usage intensity features
    df_features['Avg_Call_Duration'] = df_features['Total_Minutes'] / (df_features['Total_Calls'] + 1)
    df_features['Day_Call_Duration'] = df_features['Day Mins'] / (df_features['Day Calls'] + 1)
    df_features['Eve_Call_Duration'] = df_features['Eve Mins'] / (df_features['Eve Calls'] + 1)
    df_features['Night_Call_Duration'] = df_features['Night Mins'] / (df_features['Night Calls'] + 1)
    
    # 3. Usage pattern features (ratios)
    df_features['Day_Usage_Ratio'] = df_features['Day Mins'] / (df_features['Total_Minutes'] + 1)
    df_features['Eve_Usage_Ratio'] = df_features['Eve Mins'] / (df_features['Total_Minutes'] + 1)
    df_features['Night_Usage_Ratio'] = df_features['Night Mins'] / (df_features['Total_Minutes'] + 1)
    df_features['Intl_Usage_Ratio'] = df_features['Intl Mins'] / (df_features['Total_Minutes'] + 1)
    
    # 4. Revenue-based features
    df_features['Revenue_Per_Minute'] = df_features['Total_Charges'] / (df_features['Total_Minutes'] + 1)
    df_features['Day_Revenue_Rate'] = df_features['Day Charge'] / (df_features['Day Mins'] + 1)
    df_features['Eve_Revenue_Rate'] = df_features['Eve Charge'] / (df_features['Eve Mins'] + 1)
    df_features['Night_Revenue_Rate'] = df_features['Night Charge'] / (df_features['Night Mins'] + 1)
    
    # 5. Customer lifecycle features
    df_features['Account_Length_Months'] = df_features['Account Length'] / 30.44
    df_features['Usage_Per_Day'] = df_features['Total_Minutes'] / (df_features['Account Length'] + 1)
    df_features['Revenue_Per_Day'] = df_features['Total_Charges'] / (df_features['Account Length'] + 1)
    df_features['Service_Calls_Per_Month'] = df_features['CustServ Calls'] / (df_features['Account_Length_Months'] + 1)
    
    # 6. Behavioral indicators (binary features)
    df_features['Is_Heavy_Day_User'] = (df_features['Day Mins'] > df_features['Day Mins'].quantile(0.75)).astype(int)
    df_features['Is_Heavy_Eve_User'] = (df_features['Eve Mins'] > df_features['Eve Mins'].quantile(0.75)).astype(int)
    df_features['Is_Heavy_Night_User'] = (df_features['Night Mins'] > df_features['Night Mins'].quantile(0.75)).astype(int)
    df_features['Is_Intl_User'] = (df_features['Intl Calls'] > 0).astype(int)
    df_features['Is_High_Service_User'] = (df_features['CustServ Calls'] > 2).astype(int)
    df_features['Has_Voicemail'] = (df_features['VMail Message'] > 0).astype(int)
    df_features['Is_High_Value_Customer'] = (df_features['Total_Charges'] > df_features['Total_Charges'].quantile(0.8)).astype(int)
    
    # 7. Risk indicators
    df_features['Churn_Risk_Score'] = (
        (df_features['CustServ Calls'] > 3).astype(int) * 3 +
        (df_features['Total_Minutes'] < df_features['Total_Minutes'].quantile(0.2)).astype(int) * 2 +
        (df_features['Account Length'] < 90).astype(int) * 1
    )
    
    # Handle infinite values and NaN again after feature engineering
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    for col in df_features.select_dtypes(include=[np.number]).columns:
        if df_features[col].isnull().any():
            df_features[col].fillna(df_features[col].median(), inplace=True)
            
    # CRITICAL FIX: Reset index after all preprocessing
    df_features = df_features.reset_index(drop=True)
            
    return df_features

# --- Prediction and Recommendation Function (Reusing Logic from Phase 3) ---
def generate_recommendations(df_input_processed):
    # Ensure all required feature columns are present and in the correct order
    # It's crucial that X_predict has the exact same columns in the exact same order as feature_columns
    missing_cols = set(feature_columns) - set(df_input_processed.columns)
    if missing_cols:
        st.error(f"❌ Missing required features after preprocessing: {', '.join(missing_cols)}. Please ensure your input data can generate these features.")
        st.stop()
    
    # Align columns to the order expected by the model
    X_predict = df_input_processed[feature_columns]

    # Scale features if the model expects them (e.g., Logistic Regression)
    if isinstance(best_model, LogisticRegression):
        X_predict_scaled = scaler.transform(X_predict)
    else:
        X_predict_scaled = X_predict # Tree-based models don't need explicit scaling

    # Get churn probabilities (probability of class 1)
    churn_probabilities = best_model.predict_proba(X_predict_scaled)[:, 1]

    recommendations_list = []
    for i in range(len(df_input_processed)): 
        row = df_input_processed.iloc[i] # Access row by integer location
        churn_prob = churn_probabilities[i]
        
        # Upsell score (inverse of churn prob, higher is better for upsell)
        upsell_score = (1 - churn_prob) * 100 

        # --- Recommendation Logic (Based on Engineered Features and EDA) ---
        product_rec = "General Plan Optimization"
        expected_rev = 10 # Base revenue
        contact_time = "Flexible"
        reason = "General usage pattern"

        if row['Is_Heavy_Day_User'] == 1:
            product_rec = "Unlimited Day Plan"
            expected_rev = 25
            contact_time = "Weekday 10 AM - 4 PM"
            reason = "High daytime usage detected"
        elif row['Is_Heavy_Eve_User'] == 1:
            product_rec = "Unlimited Evening Plan"
            expected_rev = 20
            contact_time = "Weekday 6 PM - 9 PM"
            reason = "High evening usage detected"
        elif row['Is_Heavy_Night_User'] == 1:
            product_rec = "Unlimited Night Plan"
            expected_rev = 15
            contact_time = "Late Night / Weekend"
            reason = "High night usage detected"
        elif row['Is_Intl_User'] == 1 and row['Intl Mins'] > 5: # More specific for international
            product_rec = "International Calling Package"
            expected_rev = 18
            contact_time = "Weekday Morning"
            reason = "Frequent international calls"
        elif row['Is_High_Service_User'] == 1 and row['Churn_Risk_Score'] > 2:
            product_rec = "Premium Support & Plan Review (Retention Offer)"
            expected_rev = 30 # Value of retaining + potential upgrade
            contact_time = "Proactive - ASAP"
            reason = "High service calls and churn risk"
        elif row['Is_High_Value_Customer'] == 1 and upsell_score > 70:
            product_rec = "Premium Bundle Offer (e.g., Smart Home)"
            expected_rev = 40
            contact_time = "Flexible - High Value"
            reason = "High overall spending and loyalty"
        
        # Adjust expected revenue based on confidence
        expected_rev = expected_rev * (upsell_score / 100) # Scale revenue by confidence

        priority = "LOW"
        if upsell_score >= 85:
            priority = "VERY HIGH"
        elif upsell_score >= 70:
            priority = "HIGH"
        elif upsell_score >= 50:
            priority = "MEDIUM"
        
        recommendations_list.append({
            'Phone Number': row['Phone Number'],
            'Upsell Confidence': f"{upsell_score:.1f}%",
            'Recommended Product': product_rec,
            'Expected Monthly Revenue': f"${expected_rev:.2f}",
            'Priority': priority,
            'Best Contact Time': contact_time,
            'Reason': reason,
            'Current Day Mins': row['Day Mins'],
            'Current Eve Mins': row['Eve Mins'],
            'Current Night Mins': row['Night Mins'],
            'Current Intl Mins': row['Intl Mins'],
            'Current CustServ Calls': row['CustServ Calls'],
            'Churn Risk Score': row['Churn_Risk_Score']
        })
    
    return pd.DataFrame(recommendations_list)

# --- Streamlit App UI ---
st.title("🎯 AI Customer Upsell Prediction Tool")
st.markdown("---")

st.sidebar.header("Upload Customer Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")
    df_uploaded = pd.read_csv(uploaded_file, low_memory=False)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df_uploaded.head()) # Use st.dataframe for better display
    st.write(f"Total customers in uploaded file: {len(df_uploaded):,}")

    with st.spinner("🚀 Analyzing customer data and generating recommendations..."):
        # Preprocess data
        df_processed_for_prediction = preprocess_data_for_prediction(df_uploaded.copy()) # Pass a copy

        # Generate recommendations
        recommendations_df = generate_recommendations(df_processed_for_prediction)
    
    st.success("✅ Analysis Complete! Recommendations Generated.")
    st.markdown("---")

    # --- Display Key Metrics ---
    st.header("📈 Business Intelligence Summary")
    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(recommendations_df)
    high_priority_count = recommendations_df[recommendations_df['Priority'].isin(['HIGH', 'VERY HIGH'])].shape[0]
    expected_total_revenue = recommendations_df['Expected Monthly Revenue'].str.replace('$', '').astype(float).sum()
    avg_confidence = recommendations_df['Upsell Confidence'].str.replace('%', '').astype(float).mean()

    col1.metric("Total Customers Analyzed", f"{total_customers:,}")
    col2.metric("High Priority Opportunities", f"{high_priority_count:,}")
    col3.metric("Expected Monthly Revenue", f"${expected_total_revenue:,.2f}")
    col4.metric("Average Upsell Confidence", f"{avg_confidence:.1f}%")
    st.markdown("---")

    # --- Display Recommendations Table ---
    st.header("📋 Prioritized Upsell Recommendations")
    st.dataframe(recommendations_df) # Display all recommendations, not just head
    st.markdown("---")

    # --- Export Button ---
    csv_export = recommendations_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Call List (CSV)",
        data=csv_export,
        file_name="upsell_recommendations.csv",
        mime="text/csv",
    )
    st.markdown("---")

    # --- Visualizations ---
    st.header("📊 Recommendation Insights & Visualizations")

    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        # Pie chart for Priority Distribution
        st.subheader("Customer Priority Distribution")
        priority_counts = recommendations_df['Priority'].value_counts().sort_index()
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)

    with col_viz2:
        # Bar chart for Recommended Products Distribution
        st.subheader("Recommended Products Distribution")
        product_counts = recommendations_df['Recommended Product'].value_counts().head(7) # Top N products
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(x=product_counts.index, y=product_counts.values, ax=ax2, palette="viridis")
        ax2.set_xlabel("Product")
        ax2.set_ylabel("Number of Customers")
        ax2.set_title("Top Recommended Products")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
    
    st.markdown("---")

    # --- Detailed Insights (optional) ---
    st.subheader("💡 Actionable Insights")
    # Use st.expander for a cleaner, dynamic section
    with st.expander("Expand for Detailed Actionable Insights"):
        st.markdown("""
        - Customers flagged as **'VERY HIGH'** or **'HIGH'** priority have the highest predicted likelihood of accepting an upsell offer. Focus sales efforts here for maximum immediate impact.
        - **'Premium Support & Plan Review'** recommendations often target customers with higher churn risk, combining retention with an upsell opportunity.
        - Analyze the **'Recommended Product'** distribution to understand which offers are most frequently suggested by the AI, guiding marketing campaign focus.
        - Use **'Best Contact Time'** to optimize sales outreach strategies, improving conversion rates.
        - **Heavy Usage Patterns:** Customers with consistently high usage (e.g., `Day Mins` > 300) are prime candidates for unlimited plans.
        - **International Callers:** Those with significant `Intl Mins` or `Intl Calls` are ideal for international packages.
        - **Churn Risk Score:** Utilize the `Churn Risk Score` to identify customers at various risk levels for proactive retention offers.
        """)

else:
    st.info("⬆️ Please upload a CSV file to get started with customer upsell predictions.")
    st.markdown("""
    **Required CSV Columns:** `Phone Number`, `Account Length`, `VMail Message`, `Day Mins`, `Day Calls`, `Day Charge`, `Eve Mins`, `Eve Calls`, `Eve Charge`, `Night Mins`, `Night Calls`, `Night Charge`, `Intl Mins`, `Intl Calls`, `Intl Charge`, `CustServ Calls`, `Churn`
    """)
