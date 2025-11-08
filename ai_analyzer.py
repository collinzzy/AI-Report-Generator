import os
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime

try:
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.chat_models import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available - using fallback insights")

class AIAnalyzer:
    def __init__(self, model_provider="together"):
        self.model_provider = model_provider
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM with Together API"""
        if not LANGCHAIN_AVAILABLE:
            return None
            
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            # Try to get from streamlit secrets
            try:
                import streamlit as st
                api_key = st.secrets.get("TOGETHER_API_KEY", "")
            except:
                pass
        
        if not api_key:
            print("Warning: TOGETHER_API_KEY not found. Using fallback insights.")
            return None
        
        try:
            return ChatOpenAI(
                model="meta-llama/Llama-3-70b-chat-hf",
                temperature=0.1,
                openai_api_key=api_key,
                openai_api_base="https://api.together.xyz/v1",
                timeout=60,
                max_tokens=2000,
                streaming=False,
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return None
    
    def generate_insights(self, df, numeric_cols, categorical_cols):
        """Generate AI-powered insights from data"""
        
        # If LLM is not available, use fallback insights
        if self.llm is None:
            return self._generate_fallback_insights(df, numeric_cols, categorical_cols)
        
        # Create comprehensive data summary
        data_summary = self._create_data_summary(df, numeric_cols, categorical_cols)
        
        prompt = f"""
        You are an expert data analyst with 20 years of experience. Analyze the following dataset and provide 5-7 key insights in clear, actionable bullet points.
        
        DATA SUMMARY:
        {data_summary}
        
        Please provide insights focusing on:
        1. **Key Trends and Patterns**: What are the main trends in the data?
        2. **Notable Correlations**: What relationships exist between variables?
        3. **Data Quality Assessment**: Any data quality issues or interesting data characteristics?
        4. **Business Implications**: What do these findings mean for business decisions?
        5. **Opportunities and Risks**: What opportunities or risks can you identify?
        6. **Statistical Highlights**: Any surprising statistical findings?
        7. **Recommendations**: What actions should be taken based on this analysis?
        
        Format each insight as a clear, concise bullet point starting with •.
        Use simple language that business users can understand.
        Include specific numbers and percentages where relevant.
        Focus on actionable insights rather than just observations.
        """
        
        try:
            messages = [
                SystemMessage(content="""You are a senior data analyst who provides clear, actionable insights. 
                You excel at explaining complex data patterns in simple business terms. 
                You always provide specific recommendations and highlight both opportunities and risks."""),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            insights = self._parse_insights(response.content)
            return insights[:7]  # Return top 7 insights
            
        except Exception as e:
            print(f"Error generating AI insights: {str(e)}")
            return self._generate_fallback_insights(df, numeric_cols, categorical_cols)
    
    def _generate_fallback_insights(self, df, numeric_cols, categorical_cols):
        """Generate fallback insights when AI is not available"""
        insights = []
        
        # Basic data insights
        insights.append(f"• Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns")
        
        if numeric_cols:
            insights.append(f"• Found {len(numeric_cols)} numerical columns for analysis")
            # Add insight about first numeric column
            first_num_col = numeric_cols[0]
            avg_val = df[first_num_col].mean()
            insights.append(f"• Average {first_num_col}: {avg_val:,.2f}")
        
        if categorical_cols:
            insights.append(f"• Found {len(categorical_cols)} categorical columns for segmentation")
        
        # Data quality insights
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            missing_pct = (missing_total / (df.shape[0] * df.shape[1])) * 100
            insights.append(f"• Data quality: {missing_pct:.1f}% missing values detected")
        
        # Correlation insight if multiple numeric columns
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().iloc[0, 1]
            insights.append(f"• Correlation between {numeric_cols[0]} and {numeric_cols[1]}: {corr:.2f}")
        
        insights.append("• For deeper AI insights, configure your Together AI API key")
        
        return insights
    
    def generate_comparison_insights(self, df, category_col, value_col):
        """Generate insights for category comparisons"""
        
        # Calculate detailed statistics
        summary_stats = df.groupby(category_col)[value_col].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        # Calculate percentage differences from overall mean
        overall_mean = df[value_col].mean()
        summary_stats['pct_from_mean'] = ((summary_stats['mean'] - overall_mean) / overall_mean * 100).round(1)
        
        # Generate comparison text
        best_category = summary_stats['mean'].idxmax()
        worst_category = summary_stats['mean'].idxmin()
        best_value = summary_stats['mean'].max()
        worst_value = summary_stats['mean'].min()
        
        comparison_text = f"""
        Comparison Analysis for {value_col} by {category_col}:
        
        • Best performing category: {best_category} (avg: {best_value:.2f})
        • Lowest performing category: {worst_category} (avg: {worst_value:.2f})
        • Performance range: {best_value - worst_value:.2f}
        • Overall average: {overall_mean:.2f}
        
        Recommendation: Investigate why {best_category} performs significantly better and apply learnings to other categories.
        """
        
        return comparison_text
    
    def generate_recommendations(self, df):
        """Generate recommendations for further analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        recommendations = []
        
        if len(numeric_cols) >= 2:
            recommendations.append("Perform correlation analysis to identify relationships between numerical variables")
        
        if categorical_cols and numeric_cols:
            recommendations.append("Use comparison analysis to see how metrics vary across different categories")
        
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols and numeric_cols:
            recommendations.append("Explore time series analysis to identify trends and seasonal patterns")
        
        if len(numeric_cols) > 0:
            recommendations.append("Check for anomalies and outliers in numerical columns")
        
        if len(df) > 1000:
            recommendations.append("Consider segment analysis to identify different customer or product groups")
        
        return recommendations
    
    def _create_data_summary(self, df, numeric_cols, categorical_cols):
        """Create a comprehensive data summary for AI analysis"""
        
        summary = {
            "dataset_shape": f"{df.shape[0]:,} rows × {df.shape[1]} columns",
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB",
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()],
            "missing_values_summary": {
                "total_missing": int(df.isnull().sum().sum()),
                "missing_percentage": f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%",
                "columns_with_missing": [col for col in df.columns if df[col].isnull().sum() > 0]
            },
            "numeric_summary": {},
            "categorical_summary": {},
            "data_quality_notes": []
        }
        
        # Add numeric column summaries
        for col in numeric_cols:
            summary["numeric_summary"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "missing": int(df[col].isnull().sum())
            }
        
        # Add categorical column summaries
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            summary["categorical_summary"][col] = {
                "unique_values": int(df[col].nunique()),
                "top_categories": value_counts.head().to_dict(),
                "missing": int(df[col].isnull().sum())
            }
        
        # Add correlation information for numeric columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            significant_correlations = []
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.3:  # Only significant correlations
                        significant_correlations.append({
                            "variables": f"{numeric_cols[i]} vs {numeric_cols[j]}",
                            "correlation": round(corr, 3),
                            "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.5 else "weak"
                        })
            
            summary["significant_correlations"] = significant_correlations
        
        # Add data quality observations
        high_missing_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.1]
        if high_missing_cols:
            summary["data_quality_notes"].append(f"High missing values (>10%) in: {', '.join(high_missing_cols)}")
        
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            summary["data_quality_notes"].append(f"Constant values in: {', '.join(constant_cols)}")
        
        high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_cardinality_cols:
            summary["data_quality_notes"].append(f"High cardinality (>50 unique values) in: {', '.join(high_cardinality_cols)}")
        
        return json.dumps(summary, indent=2)
    
    def _parse_insights(self, ai_response):
        """Parse AI response into individual insights"""
        insights = []
        lines = ai_response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Handle different bullet point formats
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                insight = line[1:].strip()
                if len(insight) > 10:  # Only include substantial insights
                    insights.append(insight)
            elif line and re.match(r'^\d+\.', line):  # Numbered lists
                insight = re.sub(r'^\d+\.', '', line).strip()
                if len(insight) > 10:
                    insights.append(insight)
            elif line and len(line) > 50 and not line.startswith('#'):  # Substantive lines
                # Check if it looks like an insight (not a section header)
                if not line.isupper() and not line.endswith(':'):
                    insights.append(line)
        
        # If no structured insights found, split by sentences
        if not insights:
            sentences = re.split(r'[.!?]+', ai_response)
            insights = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        return insights[:10]  # Limit to 10 insights