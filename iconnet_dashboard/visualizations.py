# -*- coding: utf-8 -*-
"""
Visualization module for ICONNET Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from utils import get_color_palette, format_currency, format_percentage, handle_error
from config import COLOR_SCHEMES

class DashboardVisualizations:
    """Create visualizations for the dashboard"""
    
    def __init__(self):
        self.color_palette = get_color_palette(10)
        
    @handle_error
    def create_revenue_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create revenue distribution histogram"""
        fig = px.histogram(
            df, 
            x='total_monthly_revenue',
            title='Monthly Revenue Distribution',
            nbins=30,
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Monthly Revenue (Rp)",
            yaxis_title="Number of Customers",
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_segment_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create customer segments pie chart"""
        segment_counts = df['segment'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segments Distribution',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    @handle_error
    def create_contract_bar_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create contract duration bar chart"""
        contract_counts = df['contract_duration'].value_counts()
        
        fig = px.bar(
            x=contract_counts.index,
            y=contract_counts.values,
            title='Contract Duration Distribution',
            color_discrete_sequence=[COLOR_SCHEMES["secondary"]]
        )
        
        fig.update_layout(
            xaxis_title="Contract Duration",
            yaxis_title="Number of Customers",
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_churn_analysis_charts(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create churn analysis visualizations"""
        charts = {}
        
        # Churn by segment
        churn_by_segment = df.groupby('segment')['churn'].agg(['mean', 'count', 'sum']).reset_index()
        churn_by_segment.columns = ['segment', 'churn_rate', 'total_customers', 'churned_customers']
        
        charts['churn_by_segment'] = px.bar(
            churn_by_segment,
            x='segment',
            y='churn_rate',
            title='Churn Rate by Customer Segment',
            color='churn_rate',
            color_continuous_scale='Reds'
        )
        
        # Churn by service type
        churn_by_service = df.groupby('service_type')['churn'].agg(['mean', 'count', 'sum']).reset_index()
        churn_by_service.columns = ['service_type', 'churn_rate', 'total_customers', 'churned_customers']
        
        charts['churn_by_service'] = px.bar(
            churn_by_service,
            x='service_type',
            y='churn_rate',
            title='Churn Rate by Service Type',
            color='churn_rate',
            color_continuous_scale='Blues'
        )
        
        # Box plots for churn factors
        charts['tenure_vs_churn'] = px.box(
            df, x='churn', y='tenure',
            title='Tenure vs Churn',
            color='churn',
            color_discrete_map={0: COLOR_SCHEMES["success"], 1: COLOR_SCHEMES["danger"]}
        )
        
        charts['satisfaction_vs_churn'] = px.box(
            df, x='churn', y='customer_satisfaction',
            title='Customer Satisfaction vs Churn',
            color='churn',
            color_discrete_map={0: COLOR_SCHEMES["success"], 1: COLOR_SCHEMES["danger"]}
        )
        
        charts['downtime_vs_churn'] = px.box(
            df, x='churn', y='downtime_minutes',
            title='Downtime vs Churn',
            color='churn',
            color_discrete_map={0: COLOR_SCHEMES["success"], 1: COLOR_SCHEMES["danger"]}
        )
        
        # Update x-axis labels for box plots
        for chart_name in ['tenure_vs_churn', 'satisfaction_vs_churn', 'downtime_vs_churn']:
            charts[chart_name].update_xaxes(tickvals=[0, 1], ticktext=['No Churn', 'Churn'])
            
        return charts
    
    @handle_error
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        
        return fig
    
    @handle_error
    def create_confusion_matrix(self, cm: np.ndarray, title: str) -> go.Figure:
        """Create confusion matrix heatmap"""
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Not Churn', 'Churn'],
            y=['Not Churn', 'Churn'],
            title=title,
            color_continuous_scale='Blues'
        )
        
        return fig
    
    @handle_error
    def create_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, model_name: str) -> go.Figure:
        """Create ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.2f})',
            line=dict(color=COLOR_SCHEMES["primary"], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray, model_name: str) -> go.Figure:
        """Create Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{model_name} (AP = {avg_precision:.2f})',
            line=dict(color=COLOR_SCHEMES["secondary"], width=2)
        ))
        
        fig.update_layout(
            title=f'Precision-Recall Curve - {model_name}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_feature_importance_chart(self, feature_importance: Dict[str, float], title: str) -> go.Figure:
        """Create feature importance bar chart"""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features[:15])  # Top 15 features
        
        fig = px.bar(
            x=list(importance),
            y=list(features),
            orientation='h',
            title=title,
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_cluster_visualization(self, df: pd.DataFrame, cluster_col: str = 'cluster') -> Dict[str, go.Figure]:
        """Create cluster visualization charts"""
        charts = {}
        
        # Cluster distribution pie chart
        cluster_counts = df[cluster_col].value_counts().sort_index()
        charts['cluster_distribution'] = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title='Customer Cluster Distribution',
            color_discrete_sequence=self.color_palette
        )
        
        # Revenue by cluster
        revenue_by_cluster = df.groupby(cluster_col)['total_monthly_revenue'].agg(['mean', 'sum']).reset_index()
        charts['revenue_by_cluster'] = px.bar(
            revenue_by_cluster,
            x=cluster_col,
            y='mean',
            title='Average Revenue by Cluster',
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        # Scatter plot: Revenue vs Satisfaction colored by cluster
        charts['revenue_vs_satisfaction'] = px.scatter(
            df,
            x='customer_satisfaction',
            y='total_monthly_revenue',
            color=cluster_col,
            title='Revenue vs Customer Satisfaction by Cluster',
            color_discrete_sequence=self.color_palette
        )
        
        # Churn rate by cluster (if churn column exists)
        if 'churn' in df.columns:
            churn_by_cluster = df.groupby(cluster_col)['churn'].mean().reset_index()
            charts['churn_by_cluster'] = px.bar(
                churn_by_cluster,
                x=cluster_col,
                y='churn',
                title='Churn Rate by Cluster',
                color_discrete_sequence=[COLOR_SCHEMES["danger"]]
            )
        
        return charts
    
    @handle_error
    def create_time_series_chart(self, df: pd.DataFrame, date_col: str, value_col: str, title: str) -> go.Figure:
        """Create time series chart"""
        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            title=title,
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=value_col.replace('_', ' ').title(),
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_metrics_dashboard(self, metrics: Dict[str, Any]) -> None:
        """Create metrics dashboard with cards"""
        cols = st.columns(len(metrics))
        
        for i, (metric_name, metric_data) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(metric_data, dict):
                    value = metric_data.get('value', 0)
                    delta = metric_data.get('delta', None)
                    format_type = metric_data.get('format', 'number')
                else:
                    value = metric_data
                    delta = None
                    format_type = 'number'
                
                # Format value based on type
                if format_type == 'currency':
                    formatted_value = format_currency(value)
                elif format_type == 'percentage':
                    formatted_value = format_percentage(value)
                else:
                    formatted_value = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
                
                st.metric(
                    label=metric_name.replace('_', ' ').title(),
                    value=formatted_value,
                    delta=delta
                )
    
    @handle_error
    def create_comparison_chart(self, models_results: Dict[str, Any], metric: str = 'accuracy') -> go.Figure:
        """Create model comparison chart"""
        model_names = []
        metric_values = []
        
        for model_name, results in models_results.items():
            if 'classification_report' in results:
                model_names.append(model_name.title())
                metric_values.append(results['classification_report'][metric])
        
        fig = px.bar(
            x=model_names,
            y=metric_values,
            title=f'Model Comparison - {metric.title()}',
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        fig.update_layout(
            xaxis_title="Models",
            yaxis_title=metric.title(),
            template="plotly_white"
        )
        
        return fig