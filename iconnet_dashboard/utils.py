# -*- coding: utf-8 -*-
"""
Utility functions for ICONNET Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from config import LOGS_DIR, COLOR_SCHEMES
import warnings
warnings.filterwarnings('ignore')

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_file = LOGS_DIR / f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def handle_error(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return None
    return wrapper

def display_success(message: str):
    """Display success message with consistent styling"""
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ Success:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def display_warning(message: str):
    """Display warning message with consistent styling"""
    st.markdown(f"""
    <div class="warning-box">
        <strong>⚠️ Warning:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def display_info(message: str):
    """Display info message with consistent styling"""
    st.markdown(f"""
    <div class="info-box">
        <strong>💡 Info:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def display_error(message: str):
    """Display error message with consistent styling"""
    st.markdown(f"""
    <div class="error-box">
        <strong>❌ Error:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate dataframe has required columns"""
    if df is None or df.empty:
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        display_error(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def format_currency(amount: float, currency: str = "Rp") -> str:
    """Format currency with proper formatting"""
    return f"{currency} {amount:,.0f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage with proper formatting"""
    return f"{value:.{decimals}f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division to avoid division by zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def get_color_palette(n_colors: int, palette: str = "Set3") -> list:
    """Get color palette for visualizations"""
    import plotly.colors as pc
    
    palettes = {
        "Set3": pc.qualitative.Set3,
        "Pastel": pc.qualitative.Pastel,
        "Dark2": pc.qualitative.Dark2,
        "Bold": pc.qualitative.Bold
    }
    
    selected_palette = palettes.get(palette, pc.qualitative.Set3)
    return selected_palette[:n_colors] if n_colors <= len(selected_palette) else selected_palette

@handle_error
def export_dataframe(df: pd.DataFrame, filename: str, format: str = "csv"):
    """Export dataframe to various formats"""
    if format.lower() == "csv":
        return df.to_csv(index=False)
    elif format.lower() == "excel":
        return df.to_excel(filename, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def create_progress_tracker(steps: list) -> dict:
    """Create progress tracker for multi-step operations"""
    return {
        "steps": steps,
        "current_step": 0,
        "progress_bar": st.progress(0),
        "status_text": st.empty()
    }

def update_progress(tracker: dict, step_name: str):
    """Update progress tracker"""
    if tracker["current_step"] < len(tracker["steps"]):
        progress = (tracker["current_step"] + 1) / len(tracker["steps"])
        tracker["progress_bar"].progress(progress)
        tracker["status_text"].text(f"Step {tracker['current_step'] + 1}/{len(tracker['steps'])}: {step_name}")
        tracker["current_step"] += 1

def complete_progress(tracker: dict, success_message: str = "All steps completed successfully!"):
    """Complete progress tracking"""
    tracker["progress_bar"].progress(1.0)
    tracker["status_text"].text(success_message)
    display_success(success_message)