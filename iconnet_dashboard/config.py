# -*- coding: utf-8 -*-
"""
Configuration settings for ICONNET Dashboard
"""
import streamlit as st
from pathlib import Path
import os

# Page Configuration
PAGE_CONFIG = {
    "page_title": "ICONNET Predictive Analytics Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "SourceData"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Dashboard sections
DASHBOARD_SECTIONS = [
    "Data Overview",
    "Churn Analysis", 
    "Predictive Modeling",
    "Customer Segmentation",
    "Explainable AI",
    "Strategic Recommendations",
    "Model Management"
]

# Model parameters
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "kmeans": {
        "n_clusters": 5,
        "random_state": 42,
        "n_init": 10
    }
}

# Color schemes
COLOR_SCHEMES = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ffc107"
}

# CSS Styles
CUSTOM_CSS = """
<style>
    .main-header {font-size: 2.5rem; color: %s; font-weight: 700;}
    .sub-header {font-size: 1.8rem; color: %s; font-weight: 600;}
    .metric-label {font-size: 1.1rem; color: %s; font-weight: 500;}
    .highlight {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .footer {font-size: 0.9rem; color: #666; text-align: center; margin-top: 30px;}
    .team-info {background-color: #e8f4f8; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .logo-container {display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;}
    .logo {height: 80px; object-fit: contain;}
    .header-title {text-align: center; flex-grow: 1; margin: 0 20px;}
    .info-box {background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3; margin: 10px 0;}
    .warning-box {background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 10px 0;}
    .success-box {background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;}
    .error-box {background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin: 10px 0;}
    .stProgress .st-bo {background-color: %s;}
</style>
""" % (COLOR_SCHEMES["primary"], COLOR_SCHEMES["secondary"], COLOR_SCHEMES["success"], COLOR_SCHEMES["primary"])