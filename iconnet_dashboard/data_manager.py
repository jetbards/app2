# -*- coding: utf-8 -*-
"""
Data management module for ICONNET Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Any
from config import DATA_DIR
from utils import handle_error, display_success, display_warning, display_error, logger

class DataManager:
    """Handles data loading, generation, and validation"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.required_columns = [
            'customer_id', 'segment', 'tenure', 'contract_duration',
            'internet_speed_mbps', 'monthly_charges', 'service_type',
            'additional_services', 'monthly_usage_gb', 'downtime_minutes',
            'customer_satisfaction', 'payment_method', 'complaint_count',
            'payment_delay_days', 'churn', 'total_monthly_revenue'
        ]
    
    @handle_error
    def get_available_files(self) -> list:
        """Get list of available CSV files"""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        csv_files = [f.name for f in self.data_dir.glob("*.csv")]
        return sorted(csv_files)
    
    @handle_error
    def save_uploaded_file(self, uploaded_file) -> bool:
        """Save uploaded file to data directory"""
        try:
            file_path = self.data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            display_success(f"File {uploaded_file.name} berhasil diupload!")
            logger.info(f"File uploaded: {uploaded_file.name}")
            return True
        except Exception as e:
            display_error(f"Gagal menyimpan file: {str(e)}")
            return False
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def load_csv_file(_self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV file with error handling"""
        try:
            full_path = _self.data_dir / file_path
            if not full_path.exists():
                display_error(f"File tidak ditemukan: {file_path}")
                return None
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(full_path, encoding=encoding)
                    logger.info(f"File loaded successfully: {file_path} with encoding {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            
            display_error(f"Tidak dapat membaca file dengan encoding yang tersedia: {file_path}")
            return None
            
        except Exception as e:
            display_error(f"Error loading file {file_path}: {str(e)}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe and return validation results"""
        if df is None or df.empty:
            return {"valid": False, "message": "DataFrame kosong atau None"}
        
        validation_results = {
            "valid": True,
            "message": "Data valid",
            "missing_columns": [],
            "data_types": {},
            "missing_values": {},
            "warnings": []
        }
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            validation_results["missing_columns"] = list(missing_cols)
            validation_results["warnings"].append(f"Missing columns: {missing_cols}")
        
        # Check data types
        for col in df.columns:
            validation_results["data_types"][col] = str(df[col].dtype)
        
        # Check missing values
        missing_values = df.isnull().sum()
        validation_results["missing_values"] = {
            col: int(count) for col, count in missing_values.items() if count > 0
        }
        
        # Additional validations
        if 'churn' in df.columns:
            unique_churn = df['churn'].unique()
            if not set(unique_churn).issubset({0, 1}):
                validation_results["warnings"].append("Churn column should contain only 0 and 1")
        
        if 'customer_satisfaction' in df.columns:
            satisfaction_range = df['customer_satisfaction'].dropna()
            if len(satisfaction_range) > 0:
                min_sat, max_sat = satisfaction_range.min(), satisfaction_range.max()
                if min_sat < 1 or max_sat > 10:
                    validation_results["warnings"].append("Customer satisfaction should be between 1-10")
        
        return validation_results
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def generate_sample_data(_self, n_customers: int = 1000) -> pd.DataFrame:
        """Generate realistic sample data for ICONNET customers"""
        np.random.seed(42)
        
        logger.info(f"Generating sample data for {n_customers} customers")
        
        # Customer segments with realistic distribution
        segments = ['Residential', 'Corporate', 'Enterprise', 'Government']
        segment_probs = [0.65, 0.20, 0.10, 0.05]
        
        data = {
            'customer_id': [f"ICON{str(i).zfill(6)}" for i in range(1, n_customers+1)],
            'segment': np.random.choice(segments, n_customers, p=segment_probs),
            'tenure': np.random.randint(1, 72, n_customers),  # months
            'contract_duration': np.random.choice(['Monthly', '1 Year', '2 Years'], n_customers, p=[0.4, 0.4, 0.2]),
        }
        
        # Initialize service-specific features
        internet_speed = np.zeros(n_customers)
        monthly_charges = np.zeros(n_customers)
        service_type = []
        additional_services = []
        
        # Generate realistic data based on segment
        for i in range(n_customers):
            segment = data['segment'][i]
            
            if segment == 'Residential':
                # Residential packages (based on ICONNET products)
                speeds = [35, 50, 100]
                speed_probs = [0.4, 0.4, 0.2]
                speed = np.random.choice(speeds, p=speed_probs)
                
                # Pricing based on actual ICONNET residential packages
                price_map = {35: 265000, 50: 331000, 100: 442000}
                base_price = price_map[speed]
                
                internet_speed[i] = speed
                monthly_charges[i] = base_price * (0.9 if data['contract_duration'][i] != 'Monthly' else 1.0)
                service_type.append('Broadband')
                additional_services.append(np.random.choice(['None', 'IPTV', 'Static_IP'], p=[0.6, 0.3, 0.1]))
            
            elif segment == 'Corporate':
                # Corporate services
                services = ['Metro_Ethernet', 'IP_VPN', 'Clear_Channel']
                service = np.random.choice(services, p=[0.5, 0.3, 0.2])
                
                internet_speed[i] = np.random.choice([100, 200, 500, 1000])
                
                # Corporate pricing
                price_map = {
                    'Metro_Ethernet': np.random.uniform(8000000, 15000000),
                    'IP_VPN': np.random.uniform(3000000, 8000000),
                    'Clear_Channel': np.random.uniform(2000000, 5000000)
                }
                base_price = price_map[service]
                
                monthly_charges[i] = base_price * (0.85 if data['contract_duration'][i] == '2 Years' else 1.0)
                service_type.append(service)
                additional_services.append(np.random.choice(['Backup_Line', 'Premium_Support', 'None'], p=[0.4, 0.3, 0.3]))
            
            elif segment == 'Enterprise':
                # Enterprise managed services
                services = ['Managed_Office', 'Managed_Router', 'SD-WAN', 'Colocation']
                service = np.random.choice(services)
                
                internet_speed[i] = np.random.choice([500, 1000, 2000])
                
                # Enterprise pricing
                base_price = np.random.uniform(10000000, 25000000)
                monthly_charges[i] = base_price
                service_type.append(service)
                additional_services.append('Premium_Support')
            
            else:  # Government
                internet_speed[i] = np.random.choice([100, 200, 500])
                monthly_charges[i] = np.random.uniform(5000000, 15000000)
                service_type.append('Special_Government')
                additional_services.append('High_Security')
        
        # Add remaining features
        data.update({
            'internet_speed_mbps': internet_speed,
            'monthly_charges': monthly_charges,
            'service_type': service_type,
            'additional_services': additional_services,
            'monthly_usage_gb': np.random.lognormal(5, 1.2, n_customers),
            'downtime_minutes': np.random.exponential(120, n_customers),
            'customer_satisfaction': np.random.randint(6, 11, n_customers),
            'payment_method': np.random.choice(['Bank_Transfer', 'Credit_Card', 'Direct_Debit', 'Invoice'], n_customers),
            'complaint_count': np.random.poisson(1.5, n_customers),
            'payment_delay_days': np.random.exponential(5, n_customers),
        })
        
        df = pd.DataFrame(data)
        
        # Calculate realistic churn probability
        churn_prob = (
            (df['tenure'] < 6) * 0.3 +  # New customers
            (df['downtime_minutes'] > 300) * 0.4 +  # High downtime
            (df['customer_satisfaction'] < 7) * 0.5 +  # Low satisfaction
            (df['contract_duration'] == 'Monthly') * 0.2 +  # Monthly contracts
            (df['complaint_count'] > 3) * 0.4 +  # High complaints
            (df['payment_delay_days'] > 7) * 0.3  # Payment delays
        )
        
        # Reduce churn for enterprise customers
        churn_prob = churn_prob - (df['segment'] == 'Enterprise') * 0.3
        
        df['churn'] = np.random.binomial(1, np.clip(churn_prob, 0.05, 0.8))
        
        # Calculate total revenue including additional services
        df['total_monthly_revenue'] = df['monthly_charges'].copy()
        
        # Add revenue for additional services
        service_revenue = {
            'IPTV': 150000,
            'Static_IP': 100000,
            'Backup_Line': 500000,
            'Premium_Support': 0.2,  # 20% increase
            'High_Security': 1000000
        }
        
        for service, revenue in service_revenue.items():
            mask = df['additional_services'] == service
            if service == 'Premium_Support':
                df.loc[mask, 'total_monthly_revenue'] *= (1 + revenue)
            else:
                df.loc[mask, 'total_monthly_revenue'] += revenue
        
        logger.info(f"Sample data generated successfully: {len(df)} customers")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        if df is None or df.empty:
            return {}
        
        summary = {
            'total_customers': len(df),
            'total_revenue': df['total_monthly_revenue'].sum() if 'total_monthly_revenue' in df.columns else 0,
            'avg_revenue': df['total_monthly_revenue'].mean() if 'total_monthly_revenue' in df.columns else 0,
            'churn_rate': df['churn'].mean() * 100 if 'churn' in df.columns else 0,
            'avg_tenure': df['tenure'].mean() if 'tenure' in df.columns else 0,
            'segments': df['segment'].value_counts().to_dict() if 'segment' in df.columns else {},
            'service_types': df['service_type'].value_counts().to_dict() if 'service_type' in df.columns else {},
        }
        
        return summary