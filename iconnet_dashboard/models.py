# -*- coding: utf-8 -*-
"""
Machine Learning models module for ICONNET Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, silhouette_score
)
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config import MODEL_CONFIG, MODELS_DIR
from utils import handle_error, display_success, display_warning, display_error, logger

class ChurnPredictionModel:
    """Churn prediction model using Random Forest and XGBoost"""
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.models_dir = MODELS_DIR
        
    @handle_error
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling"""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        model_df = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['segment', 'contract_duration', 'service_type', 'additional_services', 'payment_method']
        existing_categorical = [col for col in categorical_cols if col in model_df.columns]
        
        if existing_categorical:
            model_df = pd.get_dummies(model_df, columns=existing_categorical, prefix=existing_categorical)
        
        # Define features and target
        exclude_cols = ['customer_id', 'churn', 'total_monthly_revenue']
        feature_cols = [col for col in model_df.columns if col not in exclude_cols]
        
        X = model_df[feature_cols]
        y = model_df['churn'] if 'churn' in model_df.columns else None
        
        # Store feature names
        self.feature_names = feature_cols
        
        logger.info(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")
        return X, y
    
    @handle_error
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using cluster-based undersampling"""
        if y is None:
            return X, y
            
        # Separate majority and minority classes
        majority_class_idx = y[y == 0].index
        minority_class_idx = y[y == 1].index
        
        majority_class = X.loc[majority_class_idx]
        
        if len(majority_class) > len(minority_class_idx):
            # Apply K-Means to majority class
            n_clusters = min(5, len(majority_class))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            try:
                majority_clusters = kmeans.fit_predict(majority_class)
                
                # Sample from each cluster
                sampled_indices = []
                samples_per_cluster = len(minority_class_idx) // n_clusters + 1
                
                for cluster_id in range(n_clusters):
                    cluster_mask = majority_clusters == cluster_id
                    cluster_indices = majority_class.index[cluster_mask]
                    
                    if len(cluster_indices) > 0:
                        sample_size = min(len(cluster_indices), samples_per_cluster)
                        sampled = np.random.choice(cluster_indices, sample_size, replace=False)
                        sampled_indices.extend(sampled)
                
                # Create balanced dataset
                balanced_indices = list(sampled_indices) + list(minority_class_idx)
                X_balanced = X.loc[balanced_indices]
                y_balanced = y.loc[balanced_indices]
                
                logger.info(f"Class imbalance handled: {len(X_balanced)} balanced samples")
                return X_balanced, y_balanced
                
            except Exception as e:
                logger.warning(f"Cluster-based sampling failed: {e}. Using original data.")
                return X, y
        
        return X, y
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def train_models(_self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest and XGBoost models"""
        if X is None or y is None:
            raise ValueError("Training data is None")
        
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = _self.handle_class_imbalance(X_train, y_train)
        
        # Scale features
        X_train_scaled = _self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = _self.scaler.transform(X_test)
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        _self.rf_model = RandomForestClassifier(**MODEL_CONFIG["random_forest"])
        _self.rf_model.fit(X_train_scaled, y_train_balanced)
        
        # Predictions
        y_pred_rf = _self.rf_model.predict(X_test_scaled)
        y_pred_proba_rf = _self.rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
        rf_cm = confusion_matrix(y_test, y_pred_rf)
        
        results['random_forest'] = {
            'model': _self.rf_model,
            'predictions': y_pred_rf,
            'probabilities': y_pred_proba_rf,
            'classification_report': rf_report,
            'confusion_matrix': rf_cm,
            'feature_importance': dict(zip(_self.feature_names, _self.rf_model.feature_importances_))
        }
        
        # Train XGBoost if available
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost model...")
            _self.xgb_model = XGBClassifier(**MODEL_CONFIG["xgboost"])
            _self.xgb_model.fit(X_train_scaled, y_train_balanced)
            
            # Predictions
            y_pred_xgb = _self.xgb_model.predict(X_test_scaled)
            y_pred_proba_xgb = _self.xgb_model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            xgb_report = classification_report(y_test, y_pred_xgb, output_dict=True)
            xgb_cm = confusion_matrix(y_test, y_pred_xgb)
            
            results['xgboost'] = {
                'model': _self.xgb_model,
                'predictions': y_pred_xgb,
                'probabilities': y_pred_proba_xgb,
                'classification_report': xgb_report,
                'confusion_matrix': xgb_cm,
                'feature_importance': dict(zip(_self.feature_names, _self.xgb_model.feature_importances_))
            }
        
        # Store test data for later use
        results['test_data'] = {
            'X_test': X_test,
            'y_test': y_test,
            'X_test_scaled': X_test_scaled
        }
        
        logger.info("Model training completed successfully")
        return results
    
    @handle_error
    def save_models(self, models_dict: Dict[str, Any]):
        """Save trained models to disk"""
        try:
            # Save Random Forest
            if 'random_forest' in models_dict and models_dict['random_forest']['model']:
                rf_path = self.models_dir / "random_forest_model.joblib"
                joblib.dump(models_dict['random_forest']['model'], rf_path)
                logger.info(f"Random Forest model saved to {rf_path}")
            
            # Save XGBoost
            if 'xgboost' in models_dict and models_dict['xgboost']['model']:
                xgb_path = self.models_dir / "xgboost_model.joblib"
                joblib.dump(models_dict['xgboost']['model'], xgb_path)
                logger.info(f"XGBoost model saved to {xgb_path}")
            
            # Save scaler
            scaler_path = self.models_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature names
            features_path = self.models_dir / "feature_names.joblib"
            joblib.dump(self.feature_names, features_path)
            
            display_success("Models saved successfully!")
            
        except Exception as e:
            display_error(f"Error saving models: {str(e)}")
    
    @handle_error
    def load_models(self) -> bool:
        """Load saved models from disk"""
        try:
            # Load Random Forest
            rf_path = self.models_dir / "random_forest_model.joblib"
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
                logger.info("Random Forest model loaded")
            
            # Load XGBoost
            xgb_path = self.models_dir / "xgboost_model.joblib"
            if xgb_path.exists() and XGBOOST_AVAILABLE:
                self.xgb_model = joblib.load(xgb_path)
                logger.info("XGBoost model loaded")
            
            # Load scaler
            scaler_path = self.models_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded")
            
            # Load feature names
            features_path = self.models_dir / "feature_names.joblib"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
                logger.info("Feature names loaded")
            
            return True
            
        except Exception as e:
            display_error(f"Error loading models: {str(e)}")
            return False

class CustomerSegmentationModel:
    """Customer segmentation using K-Means clustering"""
    
    def __init__(self):
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def perform_segmentation(_self, df: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
        """Perform customer segmentation"""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        # Select features for clustering
        clustering_features = [
            'tenure', 'internet_speed_mbps', 'monthly_charges',
            'monthly_usage_gb', 'downtime_minutes', 'customer_satisfaction',
            'complaint_count', 'payment_delay_days', 'total_monthly_revenue'
        ]
        
        # Filter available features
        available_features = [col for col in clustering_features if col in df.columns]
        
        if len(available_features) < 3:
            raise ValueError("Insufficient features for clustering")
        
        # Prepare data
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = _self.scaler.fit_transform(X)
        
        # Perform clustering
        _self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = _self.kmeans_model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_summary = df_clustered.groupby('cluster').agg({
            'total_monthly_revenue': ['mean', 'sum', 'count'],
            'customer_satisfaction': 'mean',
            'churn': 'mean' if 'churn' in df_clustered.columns else lambda x: 0,
            'tenure': 'mean',
            'complaint_count': 'mean'
        }).round(2)
        
        results = {
            'clustered_data': df_clustered,
            'cluster_labels': cluster_labels,
            'cluster_summary': cluster_summary,
            'silhouette_score': silhouette_avg,
            'model': _self.kmeans_model,
            'scaler': _self.scaler,
            'feature_names': available_features
        }
        
        logger.info(f"Customer segmentation completed: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
        return results

class ExplainableAI:
    """Explainable AI using LIME and SHAP"""
    
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.lime_explainer = None
        self.shap_explainer = None
        
        # Initialize LIME explainer if available
        if LIME_AVAILABLE:
            try:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=['No Churn', 'Churn'],
                    mode='classification'
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LIME explainer: {e}")
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")
    
    @handle_error
    def explain_instance_lime(self, instance_idx: int, X_test: pd.DataFrame) -> Optional[Any]:
        """Explain single instance using LIME"""
        if not LIME_AVAILABLE or self.lime_explainer is None:
            display_warning("LIME not available. Please install: pip install lime")
            return None
        
        try:
            instance = X_test.iloc[instance_idx].values
            explanation = self.lime_explainer.explain_instance(
                instance, 
                self.model.predict_proba,
                num_features=10
            )
            return explanation
        except Exception as e:
            display_error(f"Error in LIME explanation: {str(e)}")
            return None
    
    @handle_error
    def explain_instance_shap(self, instance_idx: int, X_test: pd.DataFrame) -> Optional[Any]:
        """Explain single instance using SHAP"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            display_warning("SHAP not available. Please install: pip install shap")
            return None
        
        try:
            instance = X_test.iloc[[instance_idx]]
            shap_values = self.shap_explainer.shap_values(instance)
            return shap_values
        except Exception as e:
            display_error(f"Error in SHAP explanation: {str(e)}")
            return None
    
    @handle_error
    def get_global_feature_importance(self, X_test: pd.DataFrame, max_samples: int = 100) -> Optional[Dict[str, float]]:
        """Get global feature importance using SHAP"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None
        
        try:
            # Use a sample for performance
            sample_size = min(max_samples, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_names, mean_shap_values))
            
            return feature_importance
            
        except Exception as e:
            display_error(f"Error calculating global feature importance: {str(e)}")
            return None