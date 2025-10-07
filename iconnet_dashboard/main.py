# -*- coding: utf-8 -*-
"""
Main ICONNET Predictive Analytics Dashboard
Improved version with modular architecture and better code quality
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config import PAGE_CONFIG, CUSTOM_CSS, DASHBOARD_SECTIONS
from utils import (
    setup_logging, display_success, display_warning, display_info, display_error,
    validate_dataframe, create_progress_tracker, update_progress, complete_progress
)
from data_manager import DataManager
from models import ChurnPredictionModel, CustomerSegmentationModel, ExplainableAI, LIME_AVAILABLE, SHAP_AVAILABLE
from visualizations import DashboardVisualizations

# Initialize logging
logger = setup_logging()

# Set page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

class IconnetDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.viz = DashboardVisualizations()
        self.churn_model = ChurnPredictionModel()
        self.segmentation_model = CustomerSegmentationModel()
        
    def display_header(self):
        """Display header with logos and title"""
        try:
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # PLN ICONNET Logo placeholder
                if os.path.exists("LogoPLNIconnet.jpg"):
                    st.image("LogoPLNIconnet.jpg", width=100)
                else:
                    st.markdown("""
                    <div style="text-align: center; background-color: #1a5276; color: white; padding: 20px; border-radius: 10px;">
                        <h3 style="margin: 0;">PLN ICONNET</h3>
                        <p style="margin: 0; font-size: 0.8rem;">SEMUA MAKIN MUDAH</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="header-title">
                    <h2 style="color: #1f77b4; margin: 0; text-align: center;">PEMANFAATAN ANALISIS PREDIKTIF</h2>
                    <h4 style="color: #666; margin: 0; text-align: center;">OPTIMALISASI PENDAPATAN LAYANAN ICONNET</h4>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                # BINUS Logo placeholder
                if os.path.exists("LogoBinus.png"):
                    st.image("LogoBinus.png", width=100)
                else:
                    st.markdown("""
                    <div style="text-align: center; background-color: #e74c3c; color: white; padding: 20px; border-radius: 10px;">
                        <h3 style="margin: 0;">BINUS</h3>
                        <p style="margin: 0; font-size: 0.8rem;">UNIVERSITY</p>
                    </div>
                    """, unsafe_allow_html=True)
                
        except Exception as e:
            logger.error(f"Error displaying header: {e}")
            st.markdown("## ICONNET Predictive Analytics Dashboard")

    def display_team_info(self):
        """Display team information"""
        st.markdown("""
        <div class="team-info" style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; margin: 5px 0;">
			<p style="text-align: center; margin: 2px 0; font-size: 1.2rem; font-weight: bold;">PT PLN ICON PLUS SBU REGIONAL JAWA TENGAH</p>
			<p style="text-align: center; color: #1f77b4; margin: 1px 0; font-size: 0.75rem;">Dibuat Oleh:</p>
			<div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 2px;">
				<div style="text-align: center;">
					<p style="font-weight: bold; margin: 0; color: #2c3e50; font-size: 0.75rem;">Hani Setiawan</p>
					<p style="margin: 0; color: #7f8c8d; font-size: 0.65rem;">2702464202</p>
				</div>
				<div style="text-align: center;">
					<p style="font-weight: bold; margin: 0; color: #2c3e50; font-size: 0.75rem;">Jetbar R. H. D.</p>
					<p style="margin: 0; color: #7f8c8d; font-size: 0.65rem;">2702462973</p>
				</div>
				<div style="text-align: center;">
					<p style="font-weight: bold; margin: 0; color: #2c3e50; font-size: 0.75rem;">Naufal Yafi</p>
					<p style="margin: 0; color: #7f8c8d; font-size: 0.65rem;">2702476240</p>
				</div>
			</div>
		</div>
        """, unsafe_allow_html=True)

    def load_data_section(self):
        """Handle data loading and validation"""
        st.sidebar.subheader("📂 Data Management")
        
        # Data source selection
        data_source = st.sidebar.radio(
            "Select Data Source:",
            ["Upload CSV File", "Load Existing File"]
        )
        
        df = None
        
        if data_source == "Generate Sample Data":
            n_customers = st.sidebar.slider("Number of customers:", 100, 5000, 1000, 100)
            
            if st.sidebar.button("Generate Data"):
                with st.spinner("Generating sample data..."):
                    df = self.data_manager.generate_sample_data(n_customers)
                    if df is not None:
                        st.session_state['data'] = df
                        display_success(f"Sample data generated: {len(df)} customers")
        
        elif data_source == "Upload CSV File":
            uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file is not None:
                if self.data_manager.save_uploaded_file(uploaded_file):
                    df = self.data_manager.load_csv_file(uploaded_file.name)
                    if df is not None:
                        st.session_state['data'] = df
        
        elif data_source == "Load Existing File":
            available_files = self.data_manager.get_available_files()
            
            if available_files:
                selected_file = st.sidebar.selectbox("Select file:", available_files)
                
                if st.sidebar.button("Load Data"):
                    df = self.data_manager.load_csv_file(selected_file)
                    if df is not None:
                        st.session_state['data'] = df
            else:
                display_info("No CSV files found. Please upload a file first.")
        
        # Return data from session state if available
        return st.session_state.get('data', None)

    def data_overview_section(self, df):
        """Display data overview section"""
        st.header("📋 Data Overview")
        
        # Validate data
        validation_results = self.data_manager.validate_data(df)
        
        if validation_results.get("warnings"):
            for warning in validation_results["warnings"]:
                display_warning(warning)
        
        # Data summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data info
            with st.expander("📊 Data Information"):
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.write("**Column Types:**")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Key Metrics")
            summary = self.data_manager.get_data_summary(df)
            
            # Display metrics
            metrics = {
                "Total Customers": summary.get('total_customers', 0),
                "Average Revenue": {
                    'value': summary.get('avg_revenue', 0),
                    'format': 'currency'
                },
                "Churn Rate": {
                    'value': summary.get('churn_rate', 0),
                    'format': 'percentage'
                },
                "Average Tenure": f"{summary.get('avg_tenure', 0):.1f} months"
            }
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    format_type = metric_value.get('format', 'number')
                    value = metric_value.get('value', 0)
                    
                    if format_type == 'currency':
                        st.metric(metric_name, f"Rp {value:,.0f}")
                    elif format_type == 'percentage':
                        st.metric(metric_name, f"{value:.1f}%")
                    else:
                        st.metric(metric_name, f"{value:,.0f}")
                else:
                    st.metric(metric_name, metric_value)
        
        # Visualizations
        st.subheader("📊 Data Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'total_monthly_revenue' in df.columns:
                fig_revenue = self.viz.create_revenue_distribution(df)
                st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            if 'segment' in df.columns:
                fig_segment = self.viz.create_segment_pie_chart(df)
                st.plotly_chart(fig_segment, use_container_width=True)
        
        with col3:
            if 'contract_duration' in df.columns:
                fig_contract = self.viz.create_contract_bar_chart(df)
                st.plotly_chart(fig_contract, use_container_width=True)

    def churn_analysis_section(self, df):
        """Display churn analysis section"""
        st.header("📉 Churn Analysis")
        
        if 'churn' not in df.columns:
            display_warning("Churn column not found in the data")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_churn = df['churn'].sum()
            st.metric("Total Churned", f"{total_churn:,}")
        
        with col2:
            churn_rate = df['churn'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        with col3:
            if 'total_monthly_revenue' in df.columns:
                lost_revenue = df[df['churn'] == 1]['total_monthly_revenue'].sum()
                st.metric("Lost Revenue", f"Rp {lost_revenue:,.0f}")
        
        with col4:
            avg_tenure_churned = df[df['churn'] == 1]['tenure'].mean()
            st.metric("Avg Tenure (Churned)", f"{avg_tenure_churned:.1f} months")
        
        # Churn analysis charts
        st.subheader("🔍 Churn Analysis by Segments")
        
        churn_charts = self.viz.create_churn_analysis_charts(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(churn_charts['churn_by_segment'], use_container_width=True)
            
            # Display segment data
            segment_data = df.groupby('segment')['churn'].agg(['mean', 'count', 'sum']).reset_index()
            segment_data.columns = ['Segment', 'Churn Rate', 'Total Customers', 'Churned']
            segment_data['Churn Rate'] = (segment_data['Churn Rate'] * 100).round(1).astype(str) + '%'
            st.dataframe(segment_data, use_container_width=True)
        
        with col2:
            st.plotly_chart(churn_charts['churn_by_service'], use_container_width=True)
            
            # Display service data
            service_data = df.groupby('service_type')['churn'].agg(['mean', 'count', 'sum']).reset_index()
            service_data.columns = ['Service Type', 'Churn Rate', 'Total Customers', 'Churned']
            service_data['Churn Rate'] = (service_data['Churn Rate'] * 100).round(1).astype(str) + '%'
            st.dataframe(service_data, use_container_width=True)
        
        # Churn factors analysis
        st.subheader("📊 Churn Factors Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(churn_charts['tenure_vs_churn'], use_container_width=True)
            st.plotly_chart(churn_charts['downtime_vs_churn'], use_container_width=True)
        
        with col2:
            st.plotly_chart(churn_charts['satisfaction_vs_churn'], use_container_width=True)
            
            # Correlation heatmap
            if len(df.select_dtypes(include=[np.number]).columns) > 1:
                corr_fig = self.viz.create_correlation_heatmap(df)
                st.plotly_chart(corr_fig, use_container_width=True)

    def predictive_modeling_section(self, df):
        """Display predictive modeling section"""
        st.header("🔮 Predictive Modeling")
        st.subheader("Churn Prediction using Machine Learning")
        
        if 'churn' not in df.columns:
            display_warning("Churn column not found. Cannot perform churn prediction.")
            return
        
        with st.expander("ℹ️ Model Information", expanded=False):
            st.markdown("""
            **Models Used:**
            - **Random Forest**: Ensemble method with multiple decision trees for robust predictions
            - **XGBoost**: Gradient boosting with advanced optimization (if available)
            - **Class Balancing**: Cluster-based undersampling for handling imbalanced data
            
            **Features Used:**
            - Customer demographics and behavior
            - Service usage patterns
            - Payment and satisfaction metrics
            """)
        
        # Model training button
        if st.button("🚀 Train Models", type="primary"):
            
            progress_tracker = create_progress_tracker([
                "Preparing data",
                "Handling class imbalance", 
                "Training Random Forest",
                "Training XGBoost",
                "Evaluating models"
            ])
            
            try:
                # Prepare data
                update_progress(progress_tracker, "Preparing data...")
                X, y = self.churn_model.prepare_data(df)
                
                if X is None or y is None:
                    display_error("Failed to prepare data for modeling")
                    return
                
                # Train models
                update_progress(progress_tracker, "Training models...")
                results = self.churn_model.train_models(X, y)
                
                if results:
                    complete_progress(progress_tracker, "Model training completed successfully!")
                    
                    # Store results in session state
                    st.session_state['model_results'] = results
                    st.session_state['churn_model'] = self.churn_model
                    
                    # Display results
                    self.display_model_results(results)
                    
            except Exception as e:
                display_error(f"Error in model training: {str(e)}")
                logger.error(f"Model training error: {e}")
        
        # Display existing results if available
        elif 'model_results' in st.session_state:
            self.display_model_results(st.session_state['model_results'])

    def display_model_results(self, results):
        """Display model training results"""
        st.subheader("🎯 Model Performance")
        
        # Model comparison
        col1, col2 = st.columns(2)
        
        # Random Forest Results
        with col1:
            if 'random_forest' in results:
                st.markdown("#### 🌲 Random Forest")
                rf_results = results['random_forest']
                
                # Metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Accuracy", f"{rf_results['classification_report']['accuracy']:.3f}")
                    st.metric("Precision (Churn)", f"{rf_results['classification_report']['1']['precision']:.3f}")
                with metrics_col2:
                    st.metric("Recall (Churn)", f"{rf_results['classification_report']['1']['recall']:.3f}")
                    st.metric("F1-Score (Churn)", f"{rf_results['classification_report']['1']['f1-score']:.3f}")
                
                # Confusion Matrix
                cm_fig = self.viz.create_confusion_matrix(rf_results['confusion_matrix'], "Random Forest Confusion Matrix")
                st.plotly_chart(cm_fig, use_container_width=True)
        
        # XGBoost Results
        with col2:
            if 'xgboost' in results:
                st.markdown("#### ⚡ XGBoost")
                xgb_results = results['xgboost']
                
                # Metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Accuracy", f"{xgb_results['classification_report']['accuracy']:.3f}")
                    st.metric("Precision (Churn)", f"{xgb_results['classification_report']['1']['precision']:.3f}")
                with metrics_col2:
                    st.metric("Recall (Churn)", f"{xgb_results['classification_report']['1']['recall']:.3f}")
                    st.metric("F1-Score (Churn)", f"{xgb_results['classification_report']['1']['f1-score']:.3f}")
                
                # Confusion Matrix
                cm_fig = self.viz.create_confusion_matrix(xgb_results['confusion_matrix'], "XGBoost Confusion Matrix")
                st.plotly_chart(cm_fig, use_container_width=True)
        
        # ROC Curves
        st.subheader("📈 Model Evaluation Curves")
        col1, col2 = st.columns(2)
        
        test_data = results.get('test_data', {})
        y_test = test_data.get('y_test')
        
        if y_test is not None:
            with col1:
                if 'random_forest' in results:
                    roc_fig = self.viz.create_roc_curve(
                        y_test, 
                        results['random_forest']['probabilities'], 
                        "Random Forest"
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)
            
            with col2:
                if 'xgboost' in results:
                    roc_fig = self.viz.create_roc_curve(
                        y_test, 
                        results['xgboost']['probabilities'], 
                        "XGBoost"
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)
        
        # Feature Importance
        st.subheader("🎯 Feature Importance")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'random_forest' in results:
                rf_importance_fig = self.viz.create_feature_importance_chart(
                    results['random_forest']['feature_importance'],
                    "Random Forest Feature Importance"
                )
                st.plotly_chart(rf_importance_fig, use_container_width=True)
        
        with col2:
            if 'xgboost' in results:
                xgb_importance_fig = self.viz.create_feature_importance_chart(
                    results['xgboost']['feature_importance'],
                    "XGBoost Feature Importance"
                )
                st.plotly_chart(xgb_importance_fig, use_container_width=True)
        
        # Save models button
        if st.button("💾 Save Models"):
            self.churn_model.save_models(results)

    def customer_segmentation_section(self, df):
        """Display customer segmentation section"""
        st.header("👥 Customer Segmentation")
        st.subheader("K-Means Clustering Analysis")
        
        # Segmentation parameters
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_clusters = st.slider("Number of Clusters:", 3, 8, 5)
            
            if st.button("🔍 Perform Segmentation", type="primary"):
                with st.spinner("Performing customer segmentation..."):
                    try:
                        results = self.segmentation_model.perform_segmentation(df, n_clusters)
                        
                        if results:
                            st.session_state['segmentation_results'] = results
                            display_success(f"Segmentation completed with silhouette score: {results['silhouette_score']:.3f}")
                        
                    except Exception as e:
                        display_error(f"Error in segmentation: {str(e)}")
                        logger.error(f"Segmentation error: {e}")
        
        with col2:
            # Display existing results if available
            if 'segmentation_results' in st.session_state:
                results = st.session_state['segmentation_results']
                
                st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
                st.info("Silhouette Score ranges from -1 to 1. Higher scores indicate better-defined clusters.")
        
        # Display segmentation results
        if 'segmentation_results' in st.session_state:
            results = st.session_state['segmentation_results']
            clustered_df = results['clustered_data']
            
            # Cluster visualizations
            st.subheader("📊 Cluster Analysis")
            
            cluster_charts = self.viz.create_cluster_visualization(clustered_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(cluster_charts['cluster_distribution'], use_container_width=True)
                st.plotly_chart(cluster_charts['revenue_by_cluster'], use_container_width=True)
            
            with col2:
                st.plotly_chart(cluster_charts['revenue_vs_satisfaction'], use_container_width=True)
                if 'churn_by_cluster' in cluster_charts:
                    st.plotly_chart(cluster_charts['churn_by_cluster'], use_container_width=True)
            
            # Cluster summary table
            st.subheader("📋 Cluster Summary")
            cluster_summary = results['cluster_summary']
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Detailed cluster analysis
            with st.expander("🔍 Detailed Cluster Analysis"):
                selected_cluster = st.selectbox("Select cluster for detailed analysis:", 
                                               sorted(clustered_df['cluster'].unique()))
                
                cluster_data = clustered_df[clustered_df['cluster'] == selected_cluster]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Cluster {selected_cluster} Overview:**")
                    st.write(f"- Customers: {len(cluster_data)}")
                    st.write(f"- Avg Revenue: Rp {cluster_data['total_monthly_revenue'].mean():,.0f}")
                    st.write(f"- Avg Satisfaction: {cluster_data['customer_satisfaction'].mean():.1f}")
                    if 'churn' in cluster_data.columns:
                        st.write(f"- Churn Rate: {cluster_data['churn'].mean()*100:.1f}%")
                
                with col2:
                    st.write("**Top characteristics:**")
                    # Show sample customers from this cluster
                    sample_customers = cluster_data.head()
                    st.dataframe(sample_customers[['customer_id', 'segment', 'total_monthly_revenue', 
                                                'customer_satisfaction']], use_container_width=True)

    def explainable_ai_section(self, df):
        """Display explainable AI section"""
        st.header("🔍 Explainable AI")
        st.subheader("Model Interpretability with LIME and SHAP")
        
        # Check if models are available
        if 'model_results' not in st.session_state or 'churn_model' not in st.session_state:
            display_warning("Please train the models first in the Predictive Modeling section.")
            return
        
        results = st.session_state['model_results']
        churn_model = st.session_state['churn_model']
        
        # Library availability check
        col1, col2 = st.columns(2)
        with col1:
            if LIME_AVAILABLE:
                st.success("✅ LIME is available")
            else:
                st.error("❌ LIME not available. Install with: pip install lime")
        
        with col2:
            if SHAP_AVAILABLE:
                st.success("✅ SHAP is available")
            else:
                st.error("❌ SHAP not available. Install with: pip install shap")
        
        if not (LIME_AVAILABLE or SHAP_AVAILABLE):
            display_error("Neither LIME nor SHAP is available. Please install at least one for explainable AI.")
            return
        
        # Model selection
        available_models = list(results.keys())
        if 'test_data' in available_models:
            available_models.remove('test_data')
        
        selected_model_name = st.selectbox("Select model for explanation:", available_models)
        
        if selected_model_name and selected_model_name in results:
            model = results[selected_model_name]['model']
            test_data = results['test_data']
            X_test = test_data['X_test']
            
            # Initialize explainable AI
            explainer = ExplainableAI(model, test_data['X_test_scaled'], churn_model.feature_names)
            
            # Instance explanation
            st.subheader("🎯 Single Instance Explanation")
            
            instance_idx = st.slider("Select customer instance:", 0, len(X_test)-1, 0)
            
            col1, col2 = st.columns(2)
            
            # Customer info
            with col1:
                st.write("**Customer Information:**")
                customer_info = X_test.iloc[instance_idx]
                
                # Display key customer features
                display_features = ['tenure', 'customer_satisfaction', 'monthly_charges', 'downtime_minutes']
                available_display_features = [f for f in display_features if f in customer_info.index]
                
                for feature in available_display_features:
                    st.write(f"- {feature.replace('_', ' ').title()}: {customer_info[feature]:.2f}")
                
                # Prediction
                prediction = model.predict(test_data['X_test_scaled'][instance_idx:instance_idx+1])[0]
                probability = model.predict_proba(test_data['X_test_scaled'][instance_idx:instance_idx+1])[0]
                
                st.write(f"**Prediction:** {'Churn' if prediction == 1 else 'No Churn'}")
                st.write(f"**Probability:** {probability[1]:.3f}")
            
            # Explanations
            with col2:
                if LIME_AVAILABLE:
                    if st.button("🔍 Explain with LIME"):
                        with st.spinner("Generating LIME explanation..."):
                            lime_explanation = explainer.explain_instance_lime(instance_idx, X_test)
                            
                            if lime_explanation:
                                st.success("LIME explanation generated!")
                                
                                # Extract and display explanation
                                explanation_list = lime_explanation.as_list()
                                
                                # Create explanation DataFrame
                                exp_df = pd.DataFrame(explanation_list, columns=['Feature', 'Importance'])
                                exp_df = exp_df.sort_values('Importance', key=abs, ascending=False)
                                
                                # Display as bar chart
                                import plotly.express as px
                                fig = px.bar(
                                    exp_df.head(10),
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title='LIME Feature Importance',
                                    color='Importance',
                                    color_continuous_scale='RdBu'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                if SHAP_AVAILABLE:
                    if st.button("🔍 Explain with SHAP"):
                        with st.spinner("Generating SHAP explanation..."):
                            shap_values = explainer.explain_instance_shap(instance_idx, X_test)

                            # Defensive handling: SHAP libraries may return nested arrays or objects
                            try:
                                import plotly.express as px
                                import numpy as _np

                                def _normalize_shap_values(sv, instance_idx, feature_names):
                                    """Return (arr, info) where arr is 1-D numpy array or None and info is diagnostic string."""
                                    try:
                                        # Unwrap shap.Explanation-like objects
                                        if hasattr(sv, 'values'):
                                            v = _np.asarray(sv.values)
                                        elif isinstance(sv, (list, tuple)):
                                            # Often shap returns a list per class: pick positive class (index 1) if exists
                                            if len(sv) > 1:
                                                candidate = sv[1]
                                            else:
                                                candidate = sv[0]
                                            if hasattr(candidate, 'values'):
                                                v = _np.asarray(candidate.values)
                                            else:
                                                v = _np.asarray(candidate)
                                        else:
                                            v = _np.asarray(sv)
                                    except Exception as e:
                                        return None, f"extract_error:{e}"

                                    if v is None:
                                        return None, "no_data"

                                    # Normalize numpy array
                                    v = _np.asarray(v)
                                    shape = v.shape

                                    # 1-D: already good
                                    if v.ndim == 1:
                                        return v.reshape(-1), f"shape:{shape}"

                                    # 2-D: many common formats
                                    if v.ndim == 2:
                                        # If columns match features, treat rows as instances
                                        if v.shape[1] == len(feature_names):
                                            # If there are multiple rows, pick instance row
                                            if v.shape[0] > instance_idx:
                                                return v[instance_idx].reshape(-1), f"shape:{shape},selected_row"
                                            # If only one row, use it
                                            if v.shape[0] == 1:
                                                return v[0].reshape(-1), f"shape:{shape},single_row"
                                        # If rows match features (transposed), ravel
                                        if v.shape[0] == len(feature_names) and v.shape[1] == 1:
                                            return v.ravel(), f"shape:{shape},column_vector"
                                        # If rows equal number of classes, pick class 1 if possible
                                        if v.shape[0] > 1 and v.shape[0] != len(feature_names):
                                            sel = 1 if v.shape[0] > 1 else 0
                                            row = v[sel]
                                            if row.shape[0] == len(feature_names):
                                                return row.reshape(-1), f"shape:{shape},selected_class_row"
                                        # Fallback: try to ravel
                                        return v.ravel(), f"shape:{shape},ravel_fallback"

                                    # 3-D: e.g., (classes, samples, features) or (samples, classes, features)
                                    if v.ndim == 3:
                                        # Some SHAP variants return (1, n_features, n_classes) or (n_samples, n_features, n_classes)
                                        # Normalize to (samples, features, classes) possibilities and pick class index 1
                                        # Case: (1, n_features, n_classes)
                                        if v.shape[0] == 1 and v.shape[1] == len(feature_names):
                                            # v[0] is (n_features, n_classes) -> transpose to (n_classes, n_features)
                                            tmp = v[0].T  # shape (n_classes, n_features)
                                            sel_class = 1 if tmp.shape[0] > 1 else 0
                                            row = tmp[sel_class]
                                            return row.reshape(-1), f"shape:{shape},1xf_features->class{sel_class}"

                                        # Case: (n_samples, n_features, n_classes)
                                        if v.shape[0] > 1 and v.shape[1] == len(feature_names):
                                            # pick instance row and preferred class
                                            sample = v[instance_idx]
                                            # sample shape (n_features, n_classes) -> transpose
                                            if sample.ndim == 2:
                                                tmp = sample.T  # (n_classes, n_features)
                                                sel_class = 1 if tmp.shape[0] > 1 else 0
                                                return tmp[sel_class].reshape(-1), f"shape:{shape},sample_class_features_class{sel_class}"

                                        # Case: (n_classes, n_samples, n_features)
                                        if v.shape[2] == len(feature_names) and v.shape[0] > 1 and v.shape[1] > instance_idx:
                                            sel = 1 if v.shape[0] > 1 else 0
                                            return v[sel, instance_idx].reshape(-1), f"shape:{shape},class_sample_features"

                                        # fallback: ravel
                                        return v.ravel(), f"shape:{shape},ravel3"

                                    # other: ravel as last resort
                                    return v.ravel(), f"shape:{shape},ravel_default"

                                feature_names = getattr(churn_model, 'feature_names', None) or list(X_test.columns)
                                arr, info = _normalize_shap_values(shap_values, instance_idx, feature_names)

                                if arr is None:
                                    st.error(f"Unable to parse SHAP values into a 1-D array. Diagnostic: {info}. Returned type: {type(shap_values)}")
                                else:
                                    if arr.shape[0] != len(feature_names):
                                        st.error(f"SHAP values length ({arr.shape[0]}) does not match number of features ({len(feature_names)}). Diagnostic: {info}")
                                    else:
                                        st.success("SHAP explanation generated!")

                                        # Create SHAP explanation DataFrame
                                        shap_df = pd.DataFrame({
                                            'Feature': feature_names,
                                            'SHAP_Value': arr
                                        })
                                        shap_df = shap_df.sort_values('SHAP_Value', key=abs, ascending=False)

                                        # Display as bar chart
                                        fig = px.bar(
                                            shap_df.head(10),
                                            x='SHAP_Value',
                                            y='Feature',
                                            orientation='h',
                                            title='SHAP Feature Importance',
                                            color='SHAP_Value',
                                            color_continuous_scale='RdBu'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.error(f"Error processing SHAP values: {e}")
            
            # Global explanation
            st.subheader("🌍 Global Feature Importance")
            
            if st.button("📊 Calculate Global Importance"):
                with st.spinner("Calculating global feature importance..."):
                    global_importance = explainer.get_global_feature_importance(X_test)
                    try:
                        import plotly.express as px
                        import numpy as _np

                        if global_importance is None:
                            st.error("No global feature importance returned.")
                        else:
                            feature_names = getattr(churn_model, 'feature_names', None) or list(X_test.columns)

                            # If dict, use directly
                            if isinstance(global_importance, dict):
                                items = list(global_importance.items())
                            else:
                                # Try to coerce array-like to 1-D importance vector
                                arr = _np.asarray(global_importance)

                                if arr.size == len(feature_names):
                                    flat = arr.reshape(-1)
                                elif arr.ndim >= 2 and arr.shape[-1] == len(feature_names):
                                    # take last axis as features, then if multiple entries take mean across others
                                    try:
                                        flat = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
                                    except Exception:
                                        flat = None
                                else:
                                    flat = None

                                if flat is None:
                                    st.error(f"Global importance has incompatible shape {arr.shape} for {len(feature_names)} features.")
                                    items = []
                                else:
                                    items = list(zip(feature_names, flat.tolist()))

                            if len(items) > 0:
                                importance_df = pd.DataFrame(items, columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)

                                fig = px.bar(
                                    importance_df.head(15),
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title='Global Feature Importance (SHAP)',
                                    color='Importance',
                                    color_continuous_scale='Viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Display as table
                                st.dataframe(importance_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error calculating global importance: {e}")

    def strategic_recommendations_section(self, df):
        """Display strategic recommendations section"""
        st.header("💡 Strategic Recommendations")
        st.subheader("Data-Driven Insights for Revenue Optimization")
        
        # Calculate key insights
        insights = self.calculate_business_insights(df)
        
        # Revenue optimization
        st.subheader("💰 Revenue Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### High-Value Customer Retention")
            
            high_value_threshold = df['total_monthly_revenue'].quantile(0.8)
            high_value_customers = df[df['total_monthly_revenue'] > high_value_threshold]
            high_value_churn_rate = high_value_customers['churn'].mean() * 100 if 'churn' in df.columns else 0
            
            st.write(f"- **High-value customers:** {len(high_value_customers)} ({len(high_value_customers)/len(df)*100:.1f}%)")
            st.write(f"- **Churn rate among high-value:** {high_value_churn_rate:.1f}%")
            st.write(f"- **Potential revenue at risk:** Rp {high_value_customers[high_value_customers['churn']==1]['total_monthly_revenue'].sum():,.0f}")
            
            if high_value_churn_rate > df['churn'].mean() * 100:
                st.error("⚠️ High-value customers are churning at a higher rate!")
                st.write("**Recommended Actions:**")
                st.write("• Implement dedicated account management")
                st.write("• Offer premium support services")
                st.write("• Provide exclusive benefits and discounts")
            else:
                st.success("✅ High-value customers are well retained")
        
        with col2:
            st.markdown("#### Service Optimization")
            
            # Service performance analysis
            if 'downtime_minutes' in df.columns:
                avg_downtime = df['downtime_minutes'].mean()
                high_downtime_customers = df[df['downtime_minutes'] > avg_downtime * 1.5]
                
                st.write(f"- **Average downtime:** {avg_downtime:.0f} minutes/month")
                st.write(f"- **Customers with high downtime:** {len(high_downtime_customers)}")
                
                if 'churn' in df.columns:
                    high_downtime_churn = high_downtime_customers['churn'].mean() * 100
                    st.write(f"- **Churn rate (high downtime):** {high_downtime_churn:.1f}%")
                
                st.write("**Recommended Actions:**")
                st.write("• Invest in network infrastructure upgrades")
                st.write("• Implement proactive monitoring")
                st.write("• Establish SLA guarantees")
        
        # Customer satisfaction insights
        st.subheader("😊 Customer Satisfaction")
        
        if 'customer_satisfaction' in df.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_satisfaction = df['customer_satisfaction'].mean()
                st.metric("Average Satisfaction", f"{avg_satisfaction:.1f}/10")
                
                if avg_satisfaction < 7:
                    st.error("Below target (7.5)")
                elif avg_satisfaction < 8:
                    st.warning("Needs improvement")
                else:
                    st.success("Good performance")
            
            with col2:
                low_satisfaction = df[df['customer_satisfaction'] <= 6]
                st.metric("Low Satisfaction Customers", len(low_satisfaction))
                
                if len(low_satisfaction) > 0 and 'churn' in df.columns:
                    low_sat_churn = low_satisfaction['churn'].mean() * 100
                    st.write(f"Churn rate: {low_sat_churn:.1f}%")
            
            with col3:
                high_satisfaction = df[df['customer_satisfaction'] >= 9]
                st.metric("Highly Satisfied Customers", len(high_satisfaction))
                
                if len(high_satisfaction) > 0 and 'churn' in df.columns:
                    high_sat_churn = high_satisfaction['churn'].mean() * 100
                    st.write(f"Churn rate: {high_sat_churn:.1f}%")
        
        # Segment-specific recommendations
        st.subheader("🎯 Segment-Specific Strategies")
        
        if 'segment' in df.columns:
            segments = df['segment'].unique()
            
            for segment in segments:
                segment_data = df[df['segment'] == segment]
                
                with st.expander(f"📋 {segment} Segment Strategy"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Customers:** {len(segment_data)}")
                        st.write(f"**Avg Revenue:** Rp {segment_data['total_monthly_revenue'].mean():,.0f}")
                        if 'churn' in df.columns:
                            st.write(f"**Churn Rate:** {segment_data['churn'].mean()*100:.1f}%")
                    
                    with col2:
                        # Segment-specific recommendations
                        if segment == 'Residential':
                            st.write("**Focus Areas:**")
                            st.write("• Competitive pricing for basic packages")
                            st.write("• Bundle services (Internet + IPTV)")
                            st.write("• Family-oriented marketing")
                        elif segment == 'Corporate':
                            st.write("**Focus Areas:**")
                            st.write("• Reliable business-grade services")
                            st.write("• 24/7 technical support")
                            st.write("• Scalable bandwidth options")
                        elif segment == 'Enterprise':
                            st.write("**Focus Areas:**")
                            st.write("• Dedicated account management")
                            st.write("• Custom solutions and SLAs")
                            st.write("• Advanced security features")
                        elif segment == 'Government':
                            st.write("**Focus Areas:**")
                            st.write("• Compliance with regulations")
                            st.write("• High security standards")
                            st.write("• Long-term contract incentives")
        
        # Action plan
        st.subheader("🚀 Action Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Immediate Actions (0-3 months)")
            st.write("1. **Implement churn prediction system**")
            st.write("   - Deploy trained ML models")
            st.write("   - Set up automated alerts")
            st.write("")
            st.write("2. **Launch retention campaigns**")
            st.write("   - Target high-risk customers")
            st.write("   - Offer personalized incentives")
            st.write("")
            st.write("3. **Improve customer service**")
            st.write("   - Reduce response times")
            st.write("   - Enhance technical support")
        
        with col2:
            st.markdown("#### Medium-term Actions (3-12 months)")
            st.write("1. **Network infrastructure upgrades**")
            st.write("   - Reduce downtime incidents")
            st.write("   - Improve service quality")
            st.write("")
            st.write("2. **Product portfolio optimization**")
            st.write("   - Develop new service packages")
            st.write("   - Adjust pricing strategies")
            st.write("")
            st.write("3. **Customer segmentation implementation**")
            st.write("   - Targeted marketing campaigns")
            st.write("   - Personalized service offerings")
        
        # ROI calculation
        st.subheader("💹 Expected ROI")
        
        if 'churn' in df.columns and 'total_monthly_revenue' in df.columns:
            current_churn_rate = df['churn'].mean()
            monthly_revenue_at_risk = df[df['churn'] == 1]['total_monthly_revenue'].sum()
            
            # Assume 25% reduction in churn with interventions
            improved_churn_rate = current_churn_rate * 0.75
            revenue_saved = monthly_revenue_at_risk * 0.25
            annual_revenue_saved = revenue_saved * 12
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Monthly Revenue at Risk", f"Rp {monthly_revenue_at_risk:,.0f}")
            
            with col2:
                st.metric("Potential Monthly Savings", f"Rp {revenue_saved:,.0f}")
            
            with col3:
                st.metric("Estimated Annual ROI", f"Rp {annual_revenue_saved:,.0f}")
            
            st.info(f"📊 **Assumptions:** 25% reduction in churn rate through targeted interventions")

    def calculate_business_insights(self, df):
        """Calculate business insights from data"""
        insights = {}
        
        if 'total_monthly_revenue' in df.columns:
            insights['total_revenue'] = df['total_monthly_revenue'].sum()
            insights['avg_revenue'] = df['total_monthly_revenue'].mean()
            insights['revenue_std'] = df['total_monthly_revenue'].std()
        
        if 'churn' in df.columns:
            insights['churn_rate'] = df['churn'].mean()
            insights['churned_customers'] = df['churn'].sum()
            insights['retention_rate'] = 1 - insights['churn_rate']
        
        if 'customer_satisfaction' in df.columns:
            insights['avg_satisfaction'] = df['customer_satisfaction'].mean()
            insights['low_satisfaction_customers'] = len(df[df['customer_satisfaction'] <= 6])
        
        return insights

    def model_management_section(self, df):
        """Display model management section"""
        st.header("⚙️ Model Management")
        st.subheader("Save, Load, and Export Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Operations")
            
            # Check if models exist in session
            if 'model_results' in st.session_state:
                st.success("✅ Models available in memory")
                
                if st.button("💾 Save Models to Disk"):
                    churn_model = st.session_state.get('churn_model')
                    if churn_model:
                        churn_model.save_models(st.session_state['model_results'])
                
                if st.button("📤 Export Model Performance Report"):
                    self.export_model_report(st.session_state['model_results'])
            else:
                st.warning("⚠️ No models in memory. Train models first.")
            
            # Load models from disk
            if st.button("📂 Load Models from Disk"):
                churn_model = ChurnPredictionModel()
                if churn_model.load_models():
                    st.session_state['churn_model'] = churn_model
                    display_success("Models loaded successfully!")
        
        with col2:
            st.markdown("#### Data Export")
            
            if st.button("📊 Export Current Dataset"):
                self.export_data(df)
            
            if 'segmentation_results' in st.session_state:
                if st.button("👥 Export Segmented Customers"):
                    segmented_df = st.session_state['segmentation_results']['clustered_data']
                    self.export_data(segmented_df, "segmented_customers")
        
        # Model versioning
        st.subheader("📋 Model Information")
        
        if 'model_results' in st.session_state:
            results = st.session_state['model_results']
            
            model_info = {
                "Random Forest Available": "Yes" if 'random_forest' in results else "No",
                "XGBoost Available": "Yes" if 'xgboost' in results else "No",
                "Training Date": "Current Session",
                "Feature Count": len(st.session_state.get('churn_model', ChurnPredictionModel()).feature_names or [])
            }
            
            info_df = pd.DataFrame(list(model_info.items()), columns=['Property', 'Value'])
            st.dataframe(info_df, use_container_width=True)

    def export_model_report(self, results):
        """Export model performance report"""
        try:
            report_data = []
            
            for model_name, model_results in results.items():
                if model_name != 'test_data' and 'classification_report' in model_results:
                    report = model_results['classification_report']
                    
                    report_data.append({
                        'Model': model_name.title(),
                        'Accuracy': report['accuracy'],
                        'Precision (Churn)': report['1']['precision'],
                        'Recall (Churn)': report['1']['recall'],
                        'F1-Score (Churn)': report['1']['f1-score']
                    })
            
            if report_data:
                report_df = pd.DataFrame(report_data)
                csv = report_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Model Report",
                    data=csv,
                    file_name="iconnet_model_performance_report.csv",
                    mime="text/csv"
                )
                
                display_success("Model report ready for download!")
        
        except Exception as e:
            display_error(f"Error exporting model report: {str(e)}")

    def export_data(self, df, filename_prefix="iconnet_data"):
        """Export data to CSV"""
        try:
            csv = df.to_csv(index=False)
            
            st.download_button(
                label=f"📥 Download {filename_prefix.replace('_', ' ').title()}",
                data=csv,
                file_name=f"{filename_prefix}.csv",
                mime="text/csv"
            )
            
            display_success("Data ready for download!")
        
        except Exception as e:
            display_error(f"Error exporting data: {str(e)}")

    def run(self):
        """Run the main dashboard"""
        # Display header and intro
        # self.display_header()
        self.display_team_info()
        
        # Main title
        st.title("📊 ICONNET Predictive Analytics Dashboard")
        
        # Description
        st.markdown("""
        <div class="highlight">
        Dashboard ini mendemonstrasikan bagaimana analitik prediktif dan algoritma machine learning
        (Random Forest, XGBoost, dan K-Means Clustering) dapat mengoptimalkan pendapatan layanan ICONNET
        dan meningkatkan efisiensi operasional di PT PLN ICON PLUS SBU Regional Jawa Tengah.
        </div>
        """, unsafe_allow_html=True)
        
        # Load data
        df = self.load_data_section()
        
        if df is None:
            st.warning("⚠️ Tidak ada data yang dimuat. Silakan generate sample data atau upload file CSV.")
            st.info("💡 Gunakan sidebar untuk memilih sumber data.")
            return
        
        # Navigation
        st.sidebar.header("🚀 ICONNET Analytics Navigation")
        section = st.sidebar.radio("Pilih Section:", DASHBOARD_SECTIONS)
        
        # Add info box in sidebar
        st.sidebar.markdown("""
        <div class="info-box">
        <strong>💡 Tip:</strong><br>
        Gunakan navigasi di samping untuk menjelajahi berbagai analisis prediktif yang tersedia.
        </div>
        """, unsafe_allow_html=True)
        
        # Display selected section
        try:
            if section == "Data Overview":
                self.data_overview_section(df)
            elif section == "Churn Analysis":
                self.churn_analysis_section(df)
            elif section == "Predictive Modeling":
                self.predictive_modeling_section(df)
            elif section == "Customer Segmentation":
                self.customer_segmentation_section(df)
            elif section == "Explainable AI":
                self.explainable_ai_section(df)
            elif section == "Strategic Recommendations":
                self.strategic_recommendations_section(df)
            elif section == "Model Management":
                self.model_management_section(df)
        
        except Exception as e:
            display_error(f"Error in {section}: {str(e)}")
            logger.error(f"Section error in {section}: {e}")
        
        # Footer
        st.markdown("""
        <div class="footer">
        ICONNET Predictive Analytics Dashboard - Dibuat untuk PT PLN ICON PLUS SBU Regional Jawa Tengah<br>
        Dashboard ini menggunakan teknologi Machine Learning untuk optimalisasi pendapatan dan retensi pelanggan.
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    try:
        dashboard = IconnetDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Critical error in dashboard: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()