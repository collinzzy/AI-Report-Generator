import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="AI Report Insight Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .nav-button {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.2rem;
        cursor: pointer;
        border: none;
        width: 100%;
        text-align: left;
    }
    .nav-button:hover {
        background: #1f77b4;
        color: white;
    }
    .nav-button.active {
        background: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class DataLoader:
    def load_uploaded_file(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format")
                return None
            
            # Fix date columns for Streamlit compatibility
            df = self._fix_dataframe_for_streamlit(df)
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def _fix_dataframe_for_streamlit(self, df):
        """Fix dataframe to be compatible with Streamlit's Arrow serialization"""
        df_fixed = df.copy()
        
        # Convert datetime columns to string for Streamlit compatibility
        for col in df_fixed.columns:
            if pd.api.types.is_datetime64_any_dtype(df_fixed[col]):
                df_fixed[col] = df_fixed[col].astype(str)
            # Convert other problematic types
            elif df_fixed[col].dtype == 'object':
                # Try to identify mixed types and convert to string
                try:
                    df_fixed[col] = df_fixed[col].astype(str)
                except:
                    pass
        
        return df_fixed
    
    def generate_sample_data(self, data_type):
        """Generate sample data"""
        np.random.seed(42)
        
        if data_type == "Sales Data":
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            df = pd.DataFrame({
                'date': dates.astype(str),  # Convert to string for Streamlit compatibility
                'product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'], 100),
                'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 100),
                'sales': np.random.normal(1000, 200, 100),
                'quantity': np.random.poisson(10, 100),
                'customer_rating': np.random.uniform(3.0, 5.0, 100)
            })
        elif data_type == "Financial Data":
            dates = pd.date_range('2023-01-01', periods=50, freq='W')
            df = pd.DataFrame({
                'date': dates.astype(str),  # Convert to string for Streamlit compatibility
                'revenue': np.random.normal(50000, 10000, 50),
                'expenses': np.random.normal(30000, 5000, 50),
                'profit': np.random.normal(20000, 3000, 50),
                'customers': np.random.randint(1000, 5000, 50)
            })
        else:  # Customer Data
            df = pd.DataFrame({
                'customer_id': range(1, 101),
                'age': np.random.randint(18, 70, 100),
                'gender': np.random.choice(['Male', 'Female', 'Other'], 100, p=[0.48, 0.48, 0.04]),
                'income': np.random.normal(50000, 15000, 100),
                'spending_score': np.random.randint(1, 100, 100),
                'region': np.random.choice(['Urban', 'Suburban', 'Rural'], 100)
            })
        
        return df

class AIAnalyzer:
    def generate_insights(self, df, numeric_cols, categorical_cols):
        """Generate AI-like insights"""
        insights = []
        
        # Basic insights
        insights.append(f"• Dataset contains **{len(df):,} records** with **{len(df.columns)} features**")
        
        if numeric_cols:
            insights.append(f"• **{len(numeric_cols)} numerical columns** available for quantitative analysis")
            for col in numeric_cols[:2]:  # Show first 2 columns
                avg = df[col].mean()
                std = df[col].std()
                insights.append(f"• **{col}**: Average = {avg:,.2f}, Std Dev = {std:,.2f}")
        
        if categorical_cols:
            insights.append(f"• **{len(categorical_cols)} categorical columns** for segmentation analysis")
            for col in categorical_cols[:2]:  # Show first 2 columns
                unique_count = df[col].nunique()
                top_category = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                insights.append(f"• **{col}**: {unique_count} unique values, most common: {top_category}")
        
        # Data quality insights
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            missing_pct = (missing_values / (len(df) * len(df.columns))) * 100
            insights.append(f"• **Data Quality**: {missing_pct:.1f}% missing values detected")
        else:
            insights.append("• **Data Quality**: No missing values found")
        
        # Correlation insights
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                # Find strongest correlation
                max_corr = 0
                max_pair = ("", "")
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr = abs(corr_matrix.iloc[i, j])
                        if corr > max_corr and not np.isnan(corr):
                            max_corr = corr
                            max_pair = (numeric_cols[i], numeric_cols[j])
                
                if max_corr > 0:
                    actual_corr = corr_matrix.loc[max_pair[0], max_pair[1]]
                    strength = "strong" if abs(actual_corr) > 0.7 else "moderate" if abs(actual_corr) > 0.5 else "weak"
                    insights.append(f"• **{strength.capitalize()} correlation** ({actual_corr:.2f}) between {max_pair[0]} and {max_pair[1]}")
            except Exception as e:
                pass
        
        # Date insights
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                # Try to convert back to datetime for analysis
                if df[date_col].dtype == 'object':
                    date_series = pd.to_datetime(df[date_col], errors='coerce')
                    date_range = date_series.max() - date_series.min()
                    insights.append(f"• **Time Range**: Data spans {date_range.days} days")
            except:
                pass
        
        insights.append("• **Recommendation**: Use the visualization tools below to explore patterns and relationships in your data")
        
        return insights

class VisualizationGenerator:
    def create_trend_chart(self, df, date_col, value_col):
        """Create trend chart"""
        try:
            # Handle date conversion for plotting
            plot_df = df.copy()
            if plot_df[date_col].dtype == 'object':
                plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors='coerce')
            
            fig = px.line(plot_df, x=date_col, y=value_col, 
                         title=f'📈 Trend Analysis: {value_col} over Time',
                         template='plotly_white')
            fig.update_layout(height=400, showlegend=False)
            return fig
        except Exception as e:
            # Fallback: use index as x-axis
            fig = px.line(df, y=value_col, title=f'📈 Trend: {value_col}')
            fig.update_layout(height=400, xaxis_title="Index", template='plotly_white')
            return fig
    
    def create_histogram(self, df, column):
        """Create histogram"""
        fig = px.histogram(df, x=column, 
                          title=f'📊 Distribution of {column}',
                          template='plotly_white',
                          marginal='box')
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    def create_bar_chart(self, df, category_col, value_col):
        """Create bar chart"""
        try:
            agg_df = df.groupby(category_col)[value_col].mean().reset_index()
            fig = px.bar(agg_df, x=category_col, y=value_col,
                        title=f'📊 Average {value_col} by {category_col}',
                        template='plotly_white',
                        color=value_col,
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400, showlegend=False)
            return fig
        except:
            # Fallback: simple bar chart
            fig = px.bar(df, x=category_col, y=value_col,
                        title=f'{value_col} by {category_col}',
                        template='plotly_white')
            fig.update_layout(height=400)
            return fig
    
    def create_box_plot(self, df, category_col, value_col):
        """Create box plot"""
        fig = px.box(df, x=category_col, y=value_col,
                    title=f'📦 Distribution of {value_col} by {category_col}',
                    template='plotly_white')
        fig.update_layout(height=400)
        return fig
    
    def create_scatter_plot(self, df, x_col, y_col, color_col=None):
        """Create scatter plot"""
        if color_col and color_col in df.columns:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=f'🔄 Relationship: {x_col} vs {y_col}',
                           template='plotly_white')
        else:
            fig = px.scatter(df, x=x_col, y=y_col,
                           title=f'🔄 Relationship: {x_col} vs {y_col}',
                           template='plotly_white')
        fig.update_layout(height=400)
        return fig

def main():
    # Initialize session state for navigation
    if 'current_section' not in st.session_state:
        st.session_state.current_section = "data_analysis"
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    
    # Initialize components
    data_loader = DataLoader()
    ai_analyzer = AIAnalyzer()
    viz_generator = VisualizationGenerator()
    
    # Header
    st.markdown('<div class="main-header">🤖 AI Report Insight Generator</div>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    
    # Navigation buttons
    if st.sidebar.button("📊 Data Analysis", use_container_width=True):
        st.session_state.current_section = "data_analysis"
    
    if st.sidebar.button("🤖 AI Insights", use_container_width=True):
        st.session_state.current_section = "ai_insights"
    
    if st.sidebar.button("📈 Visualizations", use_container_width=True):
        st.session_state.current_section = "visualizations"
    
    if st.sidebar.button("📤 Export", use_container_width=True):
        st.session_state.current_section = "export"
    
    st.sidebar.markdown("---")
    st.sidebar.title("📁 Data Source")
    
    # Data loading section
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Upload CSV/Excel", "Sample Data"]
    )
    
    df = st.session_state.current_df
    
    if data_source == "Upload CSV/Excel":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload your CSV or Excel file"
        )
        if uploaded_file:
            with st.spinner("Loading data..."):
                df = data_loader.load_uploaded_file(uploaded_file)
                st.session_state.current_df = df
                st.sidebar.success("Data loaded successfully!")
    
    elif data_source == "Sample Data":
        sample_type = st.sidebar.selectbox(
            "Sample Dataset",
            ["Sales Data", "Financial Data", "Customer Data"]
        )
        if st.sidebar.button("Generate Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                df = data_loader.generate_sample_data(sample_type)
                st.session_state.current_df = df
                st.sidebar.success("Sample data generated!")
    
    # Main content based on current section
    if df is not None:
        st.session_state.current_df = df
        
        if st.session_state.current_section == "data_analysis":
            show_data_analysis(df, data_loader)
        elif st.session_state.current_section == "ai_insights":
            show_ai_insights(df, ai_analyzer)
        elif st.session_state.current_section == "visualizations":
            show_visualizations(df, viz_generator)
        elif st.session_state.current_section == "export":
            show_export(df)
    else:
        show_welcome_screen(data_loader)

def show_data_analysis(df, data_loader):
    """Show data analysis section"""
    st.header("📊 Data Analysis")
    
    # Data overview
    st.subheader("📋 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Data preview
    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(df.head(10), width='stretch')
        
        tab1, tab2 = st.tabs(["Data Types", "Basic Stats"])
        with tab1:
            st.write("**Column Information**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(info_df, width='stretch')
        
        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Numerical Summary**")
                st.dataframe(df[numeric_cols].describe(), width='stretch')
            else:
                st.info("No numerical columns found for statistical summary")

def show_ai_insights(df, ai_analyzer):
    """Show AI insights section"""
    st.header("🤖 AI Insights")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if st.button("🎯 Generate AI Insights", type="primary", use_container_width=True):
        with st.spinner("Analyzing data and generating insights..."):
            insights = ai_analyzer.generate_insights(df, numeric_cols, categorical_cols)
            
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    else:
        st.info("Click the button above to generate AI-powered insights from your data")

def show_visualizations(df, viz_generator):
    """Show visualizations section"""
    st.header("📈 Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numerical columns found for visualization")
        return
    
    # Distribution Analysis
    st.subheader("📊 Distribution Analysis")
    selected_num_col = st.selectbox("Select numerical column for distribution", numeric_cols, key="dist_col")
    if selected_num_col:
        fig_hist = viz_generator.create_histogram(df, selected_num_col)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Comparison Analysis
    if categorical_cols:
        st.subheader("⚖️ Comparison Analysis")
        col1, col2 = st.columns(2)
        with col1:
            selected_cat_col = st.selectbox("Select category column", categorical_cols, key="comp_cat")
        with col2:
            selected_value_col = st.selectbox("Select value column", numeric_cols, key="comp_val")
        
        if selected_cat_col and selected_value_col:
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = viz_generator.create_bar_chart(df, selected_cat_col, selected_value_col)
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_box = viz_generator.create_box_plot(df, selected_cat_col, selected_value_col)
                st.plotly_chart(fig_box, use_container_width=True)
    
    # Trend Analysis
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        st.subheader("📈 Trend Analysis")
        trend_date_col = st.selectbox("Select date column", date_cols, key="trend_date")
        trend_value_col = st.selectbox("Select value column for trend", numeric_cols, key="trend_value")
        
        if trend_date_col and trend_value_col:
            fig_trend = viz_generator.create_trend_chart(df, trend_date_col, trend_value_col)
            st.plotly_chart(fig_trend, use_container_width=True)
    
    # Correlation Analysis
    if len(numeric_cols) >= 2:
        st.subheader("🔗 Correlation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X variable", numeric_cols, key="scatter_x")
        with col2:
            y_col = st.selectbox("Select Y variable", numeric_cols, key="scatter_y")
        
        color_col = st.selectbox("Select color variable (optional)", [None] + categorical_cols, key="scatter_color")
        
        if x_col and y_col:
            fig_scatter = viz_generator.create_scatter_plot(df, x_col, y_col, color_col)
            st.plotly_chart(fig_scatter, use_container_width=True)

def show_export(df):
    """Show export section"""
    st.header("📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Download Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="analyzed_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.subheader("Analysis Report")
        # Simple HTML report
        report_html = f"""
        <html>
        <head><title>AI Analysis Report</title></head>
        <body>
            <h1>AI Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Dataset: {df.shape[0]} rows × {df.shape[1]} columns</p>
            <h2>Summary Statistics</h2>
            {df.describe().to_html()}
        </body>
        </html>
        """
        st.download_button(
            label="📄 Download HTML Report",
            data=report_html,
            file_name="ai_analysis_report.html",
            mime="text/html",
            use_container_width=True
        )
    
    with col3:
        st.subheader("New Analysis")
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.current_df = None
            st.session_state.current_section = "data_analysis"
            st.rerun()

def show_welcome_screen(data_loader):
    """Show welcome screen when no data is loaded"""
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: #f0f2f6; border-radius: 10px;'>
        <h2 style='color: #1f77b4;'>🚀 Welcome to AI Report Insight Generator!</h2>
        <p style='font-size: 1.2rem;'>Upload your data file or use sample data to get started with AI-powered analysis.</p>
        <div style='margin-top: 2rem;'>
            <h4>🎯 Quick Start:</h4>
            <ol style='text-align: left; display: inline-block;'>
                <li>Select <b>Sample Data</b> from the sidebar</li>
                <li>Choose a dataset type</li>
                <li>Click <b>Generate Sample Data</b></li>
                <li>Use the navigation buttons to explore different sections!</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start button
    if st.button("🚀 Quick Start with Sample Data", use_container_width=True):
        df = data_loader.generate_sample_data("Sales Data")
        st.session_state.current_df = df
        st.session_state.current_section = "data_analysis"
        st.rerun()
    
    # Feature highlights
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Data Analysis</h3>
            <p>Automated statistical analysis and data profiling</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>🤖 AI Insights</h3>
            <p>Intelligent pattern detection and recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>📈 Visualizations</h3>
            <p>Interactive charts and trend analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h3>📤 Export</h3>
            <p>Download results and reports</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()