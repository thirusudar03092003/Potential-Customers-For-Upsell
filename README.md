# Potential Customers for Upsell


## 🚀 Project Overview

This project develops an **AI-powered Smart Customer Segmentation System** designed to revolutionize upselling strategies for a telecom company. Moving beyond traditional binary churn prediction, our system intelligently categorizes customers into **6 strategic segments**, enabling highly targeted and customer-respectful marketing interventions.

Our innovative approach has demonstrated **world-class performance**, achieving an astounding **2,653% ROI**, significantly surpassing the ambitious target of 1,281%. The system boasts **94.46% model accuracy** and is **100% deployment-ready**, promising transformational business impact.

## 💡 The Business Problem

Traditional telecom customer management often relies on simplistic binary churn prediction models. This leads to several critical challenges:
- **Ineffective Marketing:** Generic upselling campaigns often annoy already satisfied customers, increasing churn risk.
- **Missed Opportunities:** High-value customers showing early signs of risk don't receive timely, personalized retention efforts.
- **Resource Waste:** Untargeted marketing spends lead to low conversion rates and poor Return on Investment (ROI).
- **Customer Alienation:** A "one-size-fits-all" approach fails to recognize individual customer needs and satisfaction levels, damaging long-term relationships.

**Our Goal:** To create a sophisticated, customer-centric system that maximizes upselling revenue while simultaneously preserving customer satisfaction and loyalty.

## 🌟 Our Innovative Smart Segmentation Approach

We've engineered a paradigm shift from basic churn prediction to a multi-dimensional, customer-intelligent segmentation system.

### **6 Strategic Customer Segments:**

Our system categorizes customers into the following actionable segments:

1. **DO_NOT_DISTURB (13.6%)**: Happy, loyal customers with high satisfaction. **Strategy:** Preserve relationship, minimal contact to avoid annoyance.
2. **STANDARD_UPSELL (44.3%)**: Stable customers with clear upsell potential. **Strategy:** Standard upselling campaigns.
3. **PRIORITY_UPSELL_RETENTION (11.0%)**: High-value customers exhibiting churn risk. **Strategy:** Premium retention programs with immediate, personalized action.
4. **FIX_FIRST_THEN_UPSELL (23.6%)**: Customers experiencing service issues. **Strategy:** Resolve underlying problems before attempting any upsell.
5. **GENTLE_UPSELL (6.0%)**: Newer customers requiring a careful, nurturing approach. **Strategy:** Careful and considerate upselling.
6. **MINIMAL_CONTACT (1.5%)**: Customers with persistent issues or low potential. **Strategy:** Limited engagement and automated solutions.

This approach ensures every customer interaction is optimized for both business gain and customer experience.

## 🏆 Key Project Highlights & Achievements

Our Smart Customer Segmentation System has delivered extraordinary results:

- **Exceptional ROI:** Achieved **2,653% ROI**, significantly exceeding the ambitious target of 1,281% by **1,372 percentage points**.
- **Massive Net Benefit:** Projected **USD 9.27 Million** in net benefit for the full customer base.
- **World-Class Accuracy:** **94.46% accuracy** on the complex 6-class segmentation task.
- **Production-Ready:** Achieved a perfect **100% deployment readiness score** with robust cross-validation stability (93.55% ± 0.0005).
- **Customer-Centric AI:** Successfully balanced revenue maximization with customer relationship preservation (84.5% actionable customers, 15.5% protected loyalists).
- **Optimized Performance:** Hyperparameter optimization further enhanced model accuracy.

## ⚙️ Technical Architecture

The project follows a robust MLOps-ready structure, encompassing data processing, model development, evaluation, and a Streamlit-based user interface.

### **1. Data Pipeline & Feature Engineering**
- **Input:** Raw telecom customer data.
- **Cleaning:** Efficient handling of duplicates and missing values.
- **Feature Engineering:** Transformation of 17 raw features into 52 highly predictive engineered features (including satisfaction, value, and risk scores).
- **Output:** `data/processed/telecom_processed.csv` ready for model training.

### **2. Machine Learning Model**
- **Algorithm:** Optimized XGBoost Classifier (champion model).
- **Task:** Multi-class classification to predict the 6 strategic customer segments.
- **Performance:** 99.25% AUC capability, 94.46% accuracy.
- **Artifacts:** Trained model (`.pkl`), feature scaler (`.pkl`), and feature list (`.pkl`) are saved for deployment.

### **3. Streamlit User Interface**
- **Purpose:** Interactive dashboard for stakeholders to visualize segments, analyze business impact, and make real-time predictions.
- **Key Features:** Executive Dashboard, Customer Segment Analysis, Priority Customer Filtering, Real-time Prediction Tool, Upload & Predict New Customers, Business Impact Analysis, Model Performance Monitoring.
- **Backend:** Leverages the trained ML model and preprocessing logic for live inference.

## 📂 Project Structure

```
UPSELL_PROJECT/
├── data/
│   ├── processed/              # Cleaned datasets with 52 engineered features (60,445 customers)
│   ├── raw/                    # Original telecom dataset (101,174 records from Kaggle)
│   └── sample/                 # Test datasets for development and validation
├── models/
│   └── trained_models/         # Production-ready encoders and category mappings
├── notebooks/                  # Jupyter development environment for ML pipeline
│   ├── data/
│   │   └── processed/          # Notebook-specific processed data copies
│   ├── data_preprocessing/     # Smart segmentation pipeline (17→52 features, 6 categories)
│   ├── evaluation/             # Model validation & deployment readiness (100% criteria passed)
│   ├── exploratory_analysis/   # EDA & statistical validation of customer segments
│   ├── model_development/      # ML training & ensemble creation (94.46% accuracy achieved)
│   ├── models/                 # Trained model storage (XGBoost champion, ensembles)
│   │   └── optimized/          # Hyperparameter-optimized models (Optuna-based tuning)
│   └── outputs/
│       └── reports/            # Analysis summaries and model performance metrics
├── outputs/                    # Business deliverables and final results
│   ├── exports/
│   │   └── customer_segments/  # Segmented customer lists (6 strategic categories)
│   ├── optimization/           # Hyperparameter tuning results and best parameters
│   └── reports/                # Executive summaries & technical documentation
├── scripts/                    # Utility and automation scripts
├── src/                        # Production source code for deployment
│   ├── config/                 # Configuration files and settings
│   ├── dashboard/              # Streamlit web application (2,653% ROI visualization)
│   │   ├── components/         # Reusable dashboard UI components
│   │   └── utils/              # Dashboard utilities (data loading, business logic)
│   │       └── __pycache__/    # Python bytecode cache
│   ├── data_processing/        # Production data pipeline with feature engineering
│   ├── models/                 # Model classes and ensemble prediction utilities
│   │   └── __pycache__/        # Python bytecode cache
│   ├── prediction/             # Real-time prediction pipeline components
│   └── utils/                  # General utility functions and helpers
└── tests/                      # Quality assurance and unit testing
    ├── test_dashboard/         # Dashboard component tests
    ├── test_data_processing/   # Data pipeline validation tests
    ├── test_models/            # Model functionality and accuracy tests
    └── test_prediction/        # Prediction pipeline integration tests
```

## 🚀 Getting Started (Local Setup)

Follow these steps to set up and run the Streamlit application locally.

### **Prerequisites**

- Python 3.8+
- Git

### **1. Clone the Repository**

```bash
git clone https://github.com/mohanarajanreddy/ai-customer-upsell-prediction.git
cd ai-customer-upsell-prediction
```

### **2. Set up Virtual Environment**
It's highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### **3. Install Dependencies**
Install all required Python packages.
```bash
pip install -r requirements.txt
# If requirements.txt is not present, you can install manually:
pip install pandas numpy scikit-learn xgboost lightgbm streamlit plotly joblib optuna
```

### **4. Prepare Data and Models (Crucial Step!)**
The Streamlit app relies on preprocessed data and trained model artifacts. You MUST run the following Jupyter notebooks in order to generate these files:

1. `notebooks/data_preprocessing/01_telecom_data_preprocessing.ipynb`: This generates `data/processed/telecom_processed.csv` and necessary feature lists.
2. `notebooks/model_development/03_ensemble_model_training.ipynb`: This trains the baseline models and saves `notebooks/models/best_model_xgboost.pkl`, `notebooks/models/scaler.pkl`, `notebooks/models/feature_columns.pkl`.
3. `notebooks/model_development/05_hyperparameter_optimization.ipynb`: This runs optimization and saves `notebooks/models/optimized/xgboost_smart_segmentation.pkl` (your champion model).

Ensure these notebooks run successfully and generate their respective output files.

### **5. Run the Streamlit Application**
Navigate to the src/dashboard directory and run the app:
```bash
cd src/dashboard
streamlit run app1.py
```

Your Streamlit application will open in your web browser, typically at `http://localhost:8501` or `http://localhost:8502`.

## 📈 Streamlit Dashboard Usage

The Streamlit dashboard provides several pages for interacting with the Smart Customer Segmentation System:

- **🏆 Executive Dashboard**: High-level KPIs, overall ROI, and segment distribution.
- **👥 Customer Segments**: Detailed insights and strategies for each of the 6 customer segments.
- **⚡ Priority Customers**: Filter and analyze high-priority customers by level and segment category.
- **🔮 Prediction Tool**: Real-time prediction for individual new customer inputs.
- **⬆️ Upload & Predict New Customers**: Upload a CSV of raw customer data to get segmented predictions for an entire batch.
- **💰 Business Impact**: Detailed financial analysis and ROI breakdown.
- **📈 Model Performance**: Monitor model accuracy, stability, and comparison with baseline models.

## 💡 Future Enhancements

- **Automated Data Ingestion**: Integrate with real-time data sources (e.g., Kafka, GCS, S3).
- **MLOps Pipeline**: Implement CI/CD for automated model retraining, testing, and deployment.
- **Advanced UI/UX**: Enhance the Streamlit dashboard or transition to a more complex web framework (e.g., Flask/React).
- **A/B Testing Framework**: Integrate with marketing platforms to A/B test different segment strategies.
- **Explainable AI (XAI)**: Add SHAP or LIME for deeper model interpretability within the UI.
- **Feedback Loops**: Incorporate actual campaign results to continuously improve the model and ROI projections.

## 👥 Team Members

- Thirusudar S L
- Mohanarajan Reddy
- Nithusha Shree
- Sharmi K
- Mohammad Hajee

## 📄 License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

---

⭐ **Star this repository if you found it helpful!**
