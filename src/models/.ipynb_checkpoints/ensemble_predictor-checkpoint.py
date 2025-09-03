import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import numpy as np
import os

class EnsembleUpsellPredictor:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize all ML models with GPU acceleration where available"""
        
        # XGBoost with GPU acceleration
        self.models['xgboost'] = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='gpu_hist',  # GPU acceleration
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc'
        )
        
        # LightGBM with GPU acceleration
        self.models['lightgbm'] = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            device='gpu',  # GPU acceleration
            random_state=42,
            n_estimators=200
        )
        
        # Random Forest (CPU-based but parallel)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,  # Use all CPU cores
            random_state=42
        )
        
        # Neural Network
        self.models['neural_net'] = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all individual models"""
        
        self.initialize_models()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                if name in ['xgboost', 'lightgbm']:
                    # Tree-based models don't need scaling
                    model.fit(X_train, y_train)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # Scale for neural networks and other models
                    model.fit(X_train_scaled, y_train)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate AUC score
                auc_score = roc_auc_score(y_test, y_pred_proba)
                model_scores[name] = auc_score
                
                print(f"{name} AUC: {auc_score:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                model_scores[name] = 0.0
        
        return model_scores
    
    def create_ensemble(self, X_train, y_train):
        """Create voting ensemble of best models"""
        
        # Select models with good performance
        good_models = [(name, model) for name, model in self.models.items() 
                      if name in ['xgboost', 'lightgbm', 'random_forest']]
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=good_models,
            voting='soft',  # Use probability predictions
            n_jobs=-1
        )
        
        # Train ensemble
        print("Training ensemble model...")
        self.ensemble_model.fit(X_train, y_train)
        self.is_trained = True
        
        print("Ensemble model trained successfully!")
    
    def predict_upsell_probability(self, X):
        """Predict upsell probability using ensemble model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Get probability predictions
        upsell_probabilities = self.ensemble_model.predict_proba(X)[:, 1]
        
        return upsell_probabilities
    
    def save_models(self, model_dir='../../notebooks/models/'):  # Changed default path
        """Save all trained models"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, f'{model_dir}/{name}_model.pkl')
        
        # Save ensemble model
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, f'{model_dir}/ensemble_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir='../../notebooks/models/'):  # Changed default path
        """Load pre-trained models"""
        
        try:
            # Load ensemble model
            self.ensemble_model = joblib.load(f'{model_dir}/ensemble_model.pkl')
            
            # Load scaler
            self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
            
            self.is_trained = True
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
