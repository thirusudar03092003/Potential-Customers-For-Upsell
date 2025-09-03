import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add src to path for imports to allow relative imports from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- CORRECTED IMPORTS ---
# load_prediction_model now handles loading the model, scaler, feature_columns, and category_mapping
from utils.data_loader import load_segmentation_data, load_prediction_model 
from utils.business_logic import calculate_segment_metrics # calculate_segment_metrics is now in business_logic.py
# --- END CORRECTED IMPORTS ---

# Page configuration for a beautiful Streamlit app
st.set_page_config(
    page_title="Smart Customer Segmentation System",
    page_icon="🎯", # A target emoji for customer segmentation
    layout="wide", # Use the full width of the browser
    initial_sidebar_state="expanded" # Keep sidebar open by default
)

# Custom CSS for enhanced UI aesthetics
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); /* Gradient background */
        -webkit-background-clip: text; /* Clip background to text */
        -webkit-text-fill-color: transparent; /* Make text transparent to show gradient */
        margin-bottom: 2rem;
    }
    
    /* Styling for metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* Gradient background */
        padding: 1rem;
        border-radius: 10px;
        color: white; /* White text for contrast */
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
    }
    
    /* Styling for segment detail cards */
    .segment-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa; /* Light background */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric colors for success, warning, danger */
    .success-metric {
        color: #28a745; /* Green */
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .warning-metric {
        color: #ffc107; /* Yellow/Orange */
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .danger-metric {
        color: #dc3545; /* Red */
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Adjust Streamlit's default elements */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    .stMultiSelect > div > div {
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main Streamlit application function
def main():
    # Application Header
    st.markdown('<div class="main-header">🎯 Smart Customer Segmentation System</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Customer Intelligence for Telecom Upselling")
    
    # --- Data and Model Loading ---
    # Display a spinner while loading large datasets and models
    with st.spinner("Loading Smart Segmentation Data and Models..."):
        try:
            # Load main customer data
            customers_df = load_segmentation_data()
            
            # Load the optimized prediction model and its artifacts
            model, scaler, feature_columns, category_mapping = load_prediction_model()
            
            # Calculate segment-specific business metrics
            segment_metrics = calculate_segment_metrics(customers_df)
            
            st.success("✅ Data and models loaded successfully!")
            
        except Exception as e:
            # Display an error and stop the app if loading fails
            st.error(f"❌ Error loading data or models: {e}")
            st.info("Please ensure all data and model files exist at their expected paths and run the preprocessing and training notebooks.")
            st.stop() # Stop the app execution if critical resources are missing
    
    # --- Sidebar Navigation ---
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose Dashboard Page:",
        [
            "🏆 Executive Dashboard",
            "👥 Customer Segments", 
            "⚡ Priority Customers",
            "🔮 Prediction Tool",
            "💰 Business Impact",
            "📈 Model Performance"
        ]
    )
    
    # --- System Status in Sidebar ---
    # Display key system metrics in the sidebar for quick reference
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 System Status")
    st.sidebar.success("✅ Model: 94.46% Accuracy") # Hardcoded from your successful run
    st.sidebar.success("✅ ROI: 2,653%") # Hardcoded from your successful run
    st.sidebar.success("✅ Deployment: Ready")
    st.sidebar.info(f"📊 Total Customers: {len(customers_df):,}")
    
    # --- Page Routing ---
    # Based on sidebar selection, display the corresponding dashboard page
    if "Executive Dashboard" in page:
        show_executive_dashboard(customers_df, segment_metrics)
    elif "Customer Segments" in page:
        show_customer_segments(customers_df, segment_metrics)
    elif "Priority Customers" in page:
        # Pass category_mapping to show_priority_customers for consistent naming
        show_priority_customers(customers_df, category_mapping) 
    elif "Prediction Tool" in page:
        # Pass model, scaler, feature_columns, and category_mapping for prediction
        show_prediction_tool(model, scaler, feature_columns, category_mapping)
    elif "Business Impact" in page:
        show_business_impact(segment_metrics)
    elif "Model Performance" in page:
        show_model_performance() # This page would need model and other artifacts if it shows more details

# --- Dashboard Page Functions ---

def show_executive_dashboard(customers_df, segment_metrics):
    """
    Displays an executive summary dashboard with key performance indicators,
    segment distribution, and ROI by segment.
    """
    st.header("🏆 Executive Dashboard")
    
    # Key performance indicators (KPIs) from your successful evaluation
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>94.46%</h3>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>2,653%</h3>
            <p>ROI Achievement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>USD 9.27M</h3>
            <p>Net Benefit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calculate actionable customers based on segment_metrics
        actionable_segments = ['STANDARD_UPSELL', 'PRIORITY_UPSELL_RETENTION', 'FIX_FIRST_THEN_UPSELL', 'GENTLE_UPSELL']
        total_customers = sum(metrics['count'] for metrics in segment_metrics.values())
        actionable_count = sum(segment_metrics[s]['count'] for s in actionable_segments if s in segment_metrics)
        actionable_pct = (actionable_count / total_customers * 100) if total_customers > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{actionable_pct:.1f}%</h3>
            <p>Actionable Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Customer Segment Distribution Chart ---
    st.subheader("📊 Customer Segment Distribution")
    # Ensure 'Customer_Category' is available; use dummy if not
    if 'Customer_Category' in customers_df.columns:
        segment_counts = customers_df['Customer_Category'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Smart Customer Segmentation Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3 # Use a nice color palette
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Customer Category data not available for distribution chart.")
    
    # --- ROI by Customer Segment Chart ---
    st.subheader("💰 ROI by Customer Segment")
    
    # Extract ROI data from segment_metrics
    roi_data_for_chart = {name: metrics['roi'] for name, metrics in segment_metrics.items()}
    
    fig_roi = px.bar(
        x=list(roi_data_for_chart.keys()),
        y=list(roi_data_for_chart.values()),
        title="ROI by Customer Segment (%)",
        color=list(roi_data_for_chart.values()),
        color_continuous_scale="RdYlGn", # Green for positive, Red for negative ROI
        labels={'x': 'Customer Segment', 'y': 'ROI (%)'}
    )
    fig_roi.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_roi, use_container_width=True)

def show_customer_segments(customers_df, segment_metrics):
    """
    Provides a detailed analysis page for each customer segment,
    displaying characteristics and recommended strategies.
    """
    st.header("👥 Customer Segments Analysis")
    
    # Segment selector
    segments = customers_df['Customer_Category'].unique().tolist() if 'Customer_Category' in customers_df.columns else []
    if not segments:
        st.warning("No customer segments found in the data.")
        st.stop()
        
    selected_segment = st.selectbox("Select Segment to Analyze:", segments)
    
    # Segment details
    segment_data = customers_df[customers_df['Customer_Category'] == selected_segment]
    metrics = segment_metrics.get(selected_segment, {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 {selected_segment} Overview")
        st.metric("Total Customers", f"{metrics.get('count', 0):,}")
        st.metric("Avg Monthly Revenue", f"USD {metrics.get('avg_revenue', 0.0):.2f}")
        st.metric("Avg Account Length", f"{segment_data['Account Length'].mean():.0f} days" if 'Account Length' in segment_data.columns else "N/A")
        
    with col2:
        st.subheader("📈 Segment Characteristics")
        st.metric("Avg Satisfaction Score", f"{metrics.get('avg_satisfaction', 0.0):.2f}")
        st.metric("Avg Customer Value Score", f"{metrics.get('avg_customer_value', 0.0):.3f}") # Assuming added to segment_metrics
        st.metric("Avg Risk Score", f"{metrics.get('avg_risk', 0.0):.2f}")
    
    # Segment strategy (from your documentation)
    strategies = {
        'DO_NOT_DISTURB': 'Preserve relationship, minimal contact',
        'STANDARD_UPSELL': 'Standard upselling campaigns',
        'PRIORITY_UPSELL_RETENTION': 'Premium retention with immediate action',
        'FIX_FIRST_THEN_UPSELL': 'Resolve issues first, then upsell',
        'GENTLE_UPSELL': 'Careful approach for new customers',
        'MINIMAL_CONTACT': 'Limited engagement'
    }
    
    st.subheader("🎯 Recommended Strategy")
    st.info(f"**Strategy**: {strategies.get(selected_segment, 'Custom approach based on detailed profiling.')}")
    
    # Customer list sample
    st.subheader("📋 Customer List (Sample)")
    display_columns = [
        'Phone Number', 'Total_Charges', 'Priority_Score', 'Priority_Level',
        'Satisfaction_Score', 'Risk_Score' # Added more relevant columns
    ]
    
    # Filter to only display columns that exist in the dataframe
    existing_display_columns = [col for col in display_columns if col in segment_data.columns]
    
    if not segment_data.empty and existing_display_columns:
        st.dataframe(segment_data[existing_display_columns].head(100), use_container_width=True)
    else:
        st.info("No customer data available for this segment or display columns missing.")

def show_priority_customers(customers_df, category_mapping):
    """
    Displays priority customers with filtering options for priority level and
    specific customer categories.
    """
    st.header("⚡ Priority Customers")
    st.subheader("Filter and Analyze Key Customer Segments")
    
    # --- Filter Controls ---
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority level selector
        priority_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
        selected_priority = st.selectbox(
            "Select Priority Level:", 
            priority_levels, 
            index=1, # Default to 'HIGH'
            help="Filter customers by their overall strategic priority."
        )
    
    with col2:
        # Customer Category multi-select
        all_categories_in_data = customers_df['Customer_Category'].unique().tolist() if 'Customer_Category' in customers_df.columns else []
        
        # Default to select common upsell-focused categories
        upsell_focused_categories = [
            'STANDARD_UPSELL', 
            'PRIORITY_UPSELL_RETENTION', 
            'FIX_FIRST_THEN_UPSELL', 
            'GENTLE_UPSELL'
        ]
        
        default_selected_categories = [cat for cat in upsell_focused_categories if cat in all_categories_in_data]

        selected_categories = st.multiselect(
            "Select Customer Categories:",
            options=all_categories_in_data, # Show all categories from data
            default=default_selected_categories, # Pre-select common upsell categories
            help="Further filter customers by their specific smart segment. Use this to focus on upsell-oriented categories."
        )
    
    # --- Apply Filters ---
    if 'Priority_Level' not in customers_df.columns or 'Customer_Category' not in customers_df.columns:
        st.warning("Priority_Level or Customer_Category columns are missing in the data. Cannot filter.")
        st.stop()
        
    filtered_customers = customers_df[
        (customers_df['Priority_Level'] == selected_priority) & 
        (customers_df['Customer_Category'].isin(selected_categories))
    ]
    
    # Handle empty selection for categories gracefully
    if not selected_categories:
        st.warning("Please select at least one Customer Category to display results.")
        st.stop()

    # --- Display Metrics ---
    st.markdown("---")
    st.subheader("📊 Filtered Customer Metrics")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(filtered_customers):,}</h3>
            <p>Customers Found</p>
        </div>
        """, unsafe_allow_html=True)
    with col_metric2:
        total_revenue_filtered = filtered_customers['Total_Charges'].sum() if 'Total_Charges' in filtered_customers.columns else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <h3>USD {total_revenue_filtered:,.2f}</h3>
            <p>Total Revenue (Monthly)</p>
        </div>
        """, unsafe_allow_html=True)
    with col_metric3:
        avg_monthly_revenue_filtered = filtered_customers['Total_Charges'].mean() if 'Total_Charges' in filtered_customers.columns else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <h3>USD {avg_monthly_revenue_filtered:,.2f}</h3>
            <p>Avg Monthly Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    with col_metric4:
        avg_priority_score_filtered = filtered_customers['Priority_Score'].mean() if 'Priority_Score' in filtered_customers.columns else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_priority_score_filtered:.1f}</h3>
            <p>Avg Priority Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Display Table ---
    st.subheader(f"📋 Customers ({selected_priority} Priority & Selected Categories)")
    
    if not filtered_customers.empty:
        display_columns = [
            'Phone Number', 'Customer_Category', 'Total_Charges', 
            'Priority_Score', 'Satisfaction_Score', 'Risk_Score',
            'Avg_Call_Duration', 'CustServ Calls', 'Account Length'
        ]
        
        # Filter to only display columns that exist in the dataframe
        existing_display_columns = [col for col in display_columns if col in filtered_customers.columns]
        
        if existing_display_columns:
            # Sort by Priority_Score (descending) as it's a priority list
            st.dataframe(
                filtered_customers[existing_display_columns].sort_values('Priority_Score', ascending=False),
                use_container_width=True
            )
            
            # Download button for filtered data
            csv = filtered_customers.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Filtered Customers ({selected_priority} - {len(selected_categories)} Categories)",
                data=csv,
                file_name=f"{selected_priority.lower()}_customers_filtered.csv",
                mime="text/csv"
            )
        else:
            st.info("Selected display columns are not available in the filtered data.")
    else:
        st.info("No customers found matching the selected filters. Try adjusting your selections.")

def show_prediction_tool(model, scaler, feature_columns, category_mapping):
    """
    Provides an interactive tool for real-time customer segmentation prediction
    based on user-inputted features.
    """
    st.header("🔮 Customer Segmentation Prediction Tool")
    
    st.markdown("### Enter Customer Information:")
    
    # Ensure all model artifacts are loaded
    if model is None or scaler is None or feature_columns is None or category_mapping is None:
        st.error("Model artifacts not loaded. Please check the data_loader.py for errors.")
        st.stop()
        
    # Input form for customer features
    with st.form("customer_prediction"):
        col1, col2 = st.columns(2)
        
        # Collect input data in a dictionary
        input_data = {}
        
        # --- IMPORTANT: Ensure these input features match your model's feature_columns ---
        # You have 41 features. This UI only exposes a few. For a real model, you need to
        # handle all 41. For this demo, we'll use the ones provided and fill others with 0.
        
        with col1:
            input_data['Account Length'] = st.number_input("Account Length (days)", min_value=1, max_value=500, value=100, key="al_input")
            input_data['Total_Charges'] = st.number_input("Total Monthly Charges (USD)", min_value=0.0, max_value=200.0, value=50.0, key="tc_input")
            input_data['Total_Minutes'] = st.number_input("Total Call Minutes", min_value=0.0, max_value=1000.0, value=300.0, key="tm_input")
            input_data['CustServ Calls'] = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1, key="csc_input")
            input_data['VMail Message'] = st.number_input("Voicemail Messages", min_value=0, max_value=50, value=10, key="vm_input")
            
        with col2:
            input_data['Day_Usage_Ratio'] = st.slider("Day Usage Ratio", 0.0, 1.0, 0.4, key="dur_input")
            input_data['Eve_Usage_Ratio'] = st.slider("Evening Usage Ratio", 0.0, 1.0, 0.3, key="eur_input")
            input_data['Night_Usage_Ratio'] = st.slider("Night Usage Ratio", 0.0, 1.0, 0.3, key="nur_input")
            input_data['Intl_Usage_Ratio'] = st.slider("International Usage Ratio", 0.0, 1.0, 0.1, key="iur_input")
            input_data['Has_Voicemail'] = st.checkbox("Has Voicemail", value=True, key="hvm_input")
            # Convert bool to int for model
            input_data['Has_Voicemail'] = int(input_data['Has_Voicemail']) 
            
        submitted = st.form_submit_button("🎯 Predict Customer Segment")
        
        if submitted:
            # Create a DataFrame from input data
            customer_input_df = pd.DataFrame([input_data])
            
            # --- Handle missing features (CRITICAL for real model) ---
            # You need to ensure all 41 'feature_columns' are present.
            # For features not exposed in the UI, you must provide default, mean, or imputed values.
            for col in feature_columns:
                if col not in customer_input_df.columns:
                    # Fill with 0.0 as a safe default for numeric features
                    customer_input_df[col] = 0.0 
            
            # Ensure the order of columns matches the training data used by the model
            customer_input_df = customer_input_df[feature_columns]
            
            # Scale the input features using the loaded scaler
            scaled_features = scaler.transform(customer_input_df)
            
            # Make the actual prediction using the loaded model
            prediction_proba = model.predict_proba(scaled_features)
            prediction_id = model.predict(scaled_features)[0]
            confidence = prediction_proba[0, prediction_id]
            
            # Map prediction ID to human-readable segment name
            predicted_segment = category_mapping.get(prediction_id, f"Unknown Category {prediction_id}")
            
            # Display prediction results
            st.success(f"🎯 **Predicted Segment**: {predicted_segment}")
            st.info(f"🎲 **Confidence**: {confidence:.1%}")
            
            # Strategy recommendation based on the predicted segment
            strategies = {
                'DO_NOT_DISTURB': '🛡️ Preserve relationship, minimal contact',
                'STANDARD_UPSELL': '📈 Standard upselling campaigns',
                'PRIORITY_UPSELL_RETENTION': '🚨 Premium retention with immediate action',
                'FIX_FIRST_THEN_UPSELL': '🔧 Resolve issues first, then upsell',
                'GENTLE_UPSELL': '🤝 Careful approach for new customers',
                'MINIMAL_CONTACT': '📵 Limited engagement'
            }
            
            st.markdown(f"### 🎯 Recommended Strategy:")
            st.info(strategies.get(predicted_segment, "No specific strategy defined for this segment."))

def show_business_impact(segment_metrics):
    """
    Displays the business impact analysis, including overall ROI achievement
    and revenue impact by segment.
    """
    st.header("💰 Business Impact Analysis")
    
    # --- ROI Summary ---
    st.subheader("🎯 ROI Achievement")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Target ROI",
            value="1,281%",
            help="Original ambitious target from project documentation."
        )
    
    with col2:
        # Hardcoded from your successful run (2653.4%)
        achieved_roi = 2653.4
        st.metric(
            label="Achieved ROI", 
            value=f"{achieved_roi:.0f}%",
            delta=f"+{achieved_roi - 1281:.0f}%", # Calculate delta from target
            delta_color="normal",
            help="Actual system performance based on evaluation."
        )
    
    with col3:
        target_achievement_pct = (achieved_roi / 1281) * 100
        st.metric(
            label="Target Achievement",
            value=f"{target_achievement_pct:.0f}%",
            delta="Exceeded!", # Always "Exceeded!" since you surpassed target
            delta_color="normal"
        )
    
    # --- Revenue Impact by Segment Chart ---
    st.subheader("💵 Revenue Impact by Segment")
    
    # Prepare data for chart from segment_metrics
    chart_data = pd.DataFrame([
        {'Segment': name, 'ROI': metrics['roi'], 'Net_Benefit': metrics.get('net_benefit', 0), 'Customers': metrics['count']}
        for name, metrics in segment_metrics.items()
    ])
    
    # Ensure net_benefit is populated (if not from segment_metrics, use dummy/placeholder)
    if 'Net_Benefit' not in chart_data.columns or chart_data['Net_Benefit'].isnull().any():
        st.warning("Net Benefit data is not fully available; using placeholder values for chart.")
        # For a real app, populate 'net_benefit' in calculate_segment_metrics correctly
        # This is a very rough placeholder. You'll need to implement actual net_benefit calculation
        # in calculate_segment_metrics or load it from your evaluation JSON.
        chart_data['Net_Benefit'] = chart_data['Customers'] * chart_data['ROI'] / 100 * 100 # Dummy calculation
    
    # FIX: Use absolute values for size to handle negative ROI
    chart_data['ROI_abs'] = chart_data['ROI'].abs()
        
    fig = px.scatter(
        chart_data,
        x='Customers',
        y='Net_Benefit', 
        size='ROI_abs', # Use absolute ROI values for size
        color='Segment',
        title="Revenue Impact vs Customer Count by Segment",
        hover_data=['ROI'], # Show original ROI on hover
        labels={'Customers': 'Number of Customers', 'Net_Benefit': 'Projected Net Benefit (USD)'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance():
    """
    Displays model performance metrics and a comparison of different models
    (baseline vs. optimized).
    """
    st.header("📈 Model Performance Monitoring")
    
    # Key performance metrics from your evaluation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Accuracy", "94.46%", "+0.14%") # Optimized accuracy
    with col2:
        st.metric("Cross-Validation", "93.55%", "±0.0005") # CV mean and std
    with col3:
        st.metric("Deployment Readiness", "100%", "16/16 criteria") # Perfect score
    
    # --- Model Comparison Table ---
    st.subheader("🏆 Model Comparison")
    
    # Data from your optimization notebook
    model_data = {
        'Model': ['XGBoost Optimized', 'XGBoost Baseline', 'LightGBM Optimized'], # Excluded Ensemble as it was slightly lower
        'Accuracy': [0.9446, 0.9432, 0.9416],
        'Status': ['🥇 Champion (Optimized)', '🥈 Baseline (Exceptional)', '🥉 Alternative (Optimized)']
    }
    
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True)
    
    st.markdown("---")
    st.info("The Optimized XGBoost model shows the highest accuracy and is recommended for deployment.")

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()

