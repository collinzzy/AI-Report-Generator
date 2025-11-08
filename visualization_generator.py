import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
import plotly.subplots as sp

class VisualizationGenerator:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_trend_chart(self, df, date_col, value_col, theme="plotly_white"):
        """Create an interactive trend line chart"""
        try:
            # Sort by date to ensure proper trend line
            df_sorted = df.sort_values(date_col).copy()
            
            fig = px.line(
                df_sorted, 
                x=date_col, 
                y=value_col,
                title=f'📈 Trend Analysis: {value_col} over Time',
                template=theme
            )
            
            # Add rolling average for smoother trend
            if len(df_sorted) > 7:
                window = min(30, len(df_sorted) // 10)
                df_sorted['rolling_avg'] = df_sorted[value_col].rolling(window=window, center=True).mean()
                fig.add_scatter(
                    x=df_sorted[date_col], 
                    y=df_sorted['rolling_avg'],
                    mode='lines',
                    line=dict(width=3, color='red'),
                    name=f'{window}-period Moving Average'
                )
            
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title=value_col,
                hovermode='x unified',
                showlegend=True,
                height=500
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_plot(f"Error creating trend chart: {str(e)}")
    
    def create_histogram(self, df, column, theme="plotly_white"):
        """Create histogram with distribution analysis"""
        try:
            fig = px.histogram(
                df, 
                x=column,
                title=f'📊 Distribution of {column}',
                template=theme,
                marginal='box',  # Add box plot on top
                opacity=0.7
            )
            
            # Add mean and median lines
            mean_val = df[column].mean()
            median_val = df[column].median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                         annotation_text=f"Median: {median_val:.2f}")
            
            fig.update_layout(
                xaxis_title=column,
                yaxis_title='Frequency',
                showlegend=False,
                height=400
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_plot(f"Error creating histogram: {str(e)}")
    
    def create_bar_chart(self, df, category_col, value_col, theme="plotly_white"):
        """Create bar chart for categorical comparison"""
        try:
            # Aggregate data
            agg_df = df.groupby(category_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
            agg_df = agg_df.sort_values('mean', ascending=False)
            
            fig = px.bar(
                agg_df, 
                x=category_col, 
                y='mean',
                error_y='std' if agg_df['std'].notna().all() else None,
                title=f'📊 Average {value_col} by {category_col}',
                template=theme,
                color='mean',
                color_continuous_scale='Viridis'
            )
            
            # Add count annotations
            for i, row in agg_df.iterrows():
                fig.add_annotation(
                    x=row[category_col],
                    y=row['mean'] + (row['std'] if pd.notna(row['std']) else 0),
                    text=f"n={row['count']}",
                    showarrow=False,
                    yshift=10
                )
            
            fig.update_layout(
                xaxis_title=category_col,
                yaxis_title=f'Average {value_col}',
                showlegend=False,
                height=500
            )
            
            fig.update_xaxis(tickangle=45)
            
            return fig
            
        except Exception as e:
            return self._create_error_plot(f"Error creating bar chart: {str(e)}")
    
    def create_box_plot(self, df, category_col, value_col, theme="plotly_white"):
        """Create box plot for distribution comparison"""
        try:
            fig = px.box(
                df, 
                x=category_col, 
                y=value_col,
                title=f'📦 Distribution of {value_col} by {category_col}',
                template=theme,
                color=category_col
            )
            
            fig.update_layout(
                xaxis_title=category_col,
                yaxis_title=value_col,
                showlegend=False,
                height=500
            )
            
            fig.update_xaxis(tickangle=45)
            
            return fig
            
        except Exception as e:
            return self._create_error_plot(f"Error creating box plot: {str(e)}")
    
    def create_correlation_heatmap(self, df, numeric_cols, theme="plotly_white"):
        """Create correlation heatmap for numerical variables"""
        try:
            corr_matrix = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                hoverongaps=False,
                text=corr_matrix.round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
            ))
            
            fig.update_layout(
                title='🔗 Correlation Matrix Heatmap',
                xaxis_title='Variables',
                yaxis_title='Variables',
                template=theme,
                height=600
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_plot(f"Error creating correlation heatmap: {str(e)}")
    
    def create_scatter_plot(self, df, x_col, y_col, color_col=None, theme="plotly_white"):
        """Create scatter plot for relationship analysis"""
        try:
            if color_col and color_col in df.columns:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col,
                    color=color_col,
                    title=f'🔄 Relationship: {x_col} vs {y_col}',
                    template=theme,
                    trendline='lowess'
                )
            else:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col,
                    title=f'🔄 Relationship: {x_col} vs {y_col}',
                    template=theme,
                    trendline='lowess'
                )
            
            # Calculate correlation
            correlation = df[[x_col, y_col]].corr().iloc[0, 1]
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                annotations=[
                    dict(
                        x=0.02, y=0.98,
                        xref="paper", yref="paper",
                        text=f"Correlation: {correlation:.2f}",
                        showarrow=False,
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1
                    )
                ],
                height=500
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_plot(f"Error creating scatter plot: {str(e)}")
    
    def detect_anomalies(self, df, column, method='iqr', threshold=1.5):
        """Detect anomalies in numerical data using multiple methods"""
        try:
            data = df[column].dropna()
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                anomalies_mask = z_scores > threshold
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                anomalies_mask = (data < lower_bound) | (data > upper_bound)
            else:  # percentile
                lower_bound = data.quantile(0.01)
                upper_bound = data.quantile(0.99)
                anomalies_mask = (data < lower_bound) | (data > upper_bound)
            
            anomalies = df[df[column].isin(data[anomalies_mask])]
            return anomalies
            
        except Exception as e:
            print(f"Error detecting anomalies: {str(e)}")
            return pd.DataFrame()
    
    def plot_with_anomalies(self, df, column, anomalies, theme="plotly_white"):
        """Plot data with anomalies highlighted"""
        try:
            fig = go.Figure()
            
            # Add normal data points
            normal_data = df[~df.index.isin(anomalies.index)]
            fig.add_trace(go.Scatter(
                x=normal_data.index,
                y=normal_data[column],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))
            
            # Add anomalies
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies[column],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='x-thin', line=dict(width=2))
            ))
            
            fig.update_layout(
                title=f'🚨 {column} with Anomalies Highlighted',
                xaxis_title='Index',
                yaxis_title=column,
                template=theme,
                showlegend=True,
                height=500
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_plot(f"Error creating anomaly plot: {str(e)}")
    
    def create_subplot_dashboard(self, df, numeric_cols, theme="plotly_white"):
        """Create a dashboard with multiple subplots"""
        try:
            if len(numeric_cols) == 0:
                return self._create_error_plot("No numerical columns for dashboard")
            
            # Create subplots
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Distribution Overview', 
                    'Box Plot Comparison',
                    'Trend Analysis', 
                    'Correlation Heatmap'
                ],
                specs=[
                    [{"type": "histogram"}, {"type": "box"}],
                    [{"type": "scatter"}, {"type": "heatmap"}]
                ]
            )
            
            # Histogram for first numeric column
            if len(numeric_cols) >= 1:
                fig.add_trace(
                    go.Histogram(x=df[numeric_cols[0]], name=numeric_cols[0]),
                    row=1, col=1
                )
            
            # Box plot for first few numeric columns
            if len(numeric_cols) >= 2:
                for i, col in enumerate(numeric_cols[:3]):  # Max 3 columns for clarity
                    fig.add_trace(
                        go.Box(y=df[col], name=col),
                        row=1, col=2
                    )
            
            # Scatter plot for first two numeric columns
            if len(numeric_cols) >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=df[numeric_cols[0]], 
                        y=df[numeric_cols[1]],
                        mode='markers',
                        name=f'{numeric_cols[0]} vs {numeric_cols[1]}'
                    ),
                    row=2, col=1
                )
            
            # Correlation heatmap
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmin=-1,
                        zmax=1
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title_text="📊 Comprehensive Data Dashboard",
                template=theme,
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_plot(f"Error creating dashboard: {str(e)}")
    
    def _create_error_plot(self, error_message):
        """Create a placeholder plot when visualization fails"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ Visualization Error<br>{error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            height=300
        )
        return fig