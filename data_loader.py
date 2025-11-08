import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
    
    def load_uploaded_file(self, uploaded_file):
        """Load data from uploaded CSV or Excel file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic data cleaning
            df = self.clean_data(df)
            return df
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def load_from_api(self, api_url):
        """Load data from API endpoint"""
        try:
            headers = {
                'User-Agent': 'AI-Report-Generator/1.0',
                'Accept': 'application/json'
            }
            
            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Try to parse as JSON
            data = response.json()
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Handle different JSON structures
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'results' in data:
                    df = pd.DataFrame(data['results'])
                else:
                    # Flatten nested JSON
                    df = pd.json_normalize(data)
            else:
                raise ValueError("Unsupported API response format")
            
            df = self.clean_data(df)
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing API response: {str(e)}")
    
    def generate_sample_data(self, data_type):
        """Generate sample data for demonstration purposes"""
        np.random.seed(42)  # For reproducible results
        
        if data_type == "Sales Data":
            return self._generate_sales_data()
        elif data_type == "Financial Data":
            return self._generate_financial_data()
        elif data_type == "Customer Data":
            return self._generate_customer_data()
        elif data_type == "Time Series Data":
            return self._generate_timeseries_data()
        else:
            return self._generate_sales_data()
    
    def _generate_sales_data(self):
        """Generate sample sales data"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_records = len(dates)
        
        data = {
            'date': dates,
            'product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'], n_records),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_records),
            'sales_amount': np.random.normal(1000, 300, n_records).round(2),
            'quantity': np.random.poisson(15, n_records),
            'customer_rating': np.random.uniform(3.0, 5.0, n_records).round(1),
            'category': np.random.choice(['Electronics', 'Accessories', 'Gadgets'], n_records)
        }
        
        # Add some seasonality and trend
        data['sales_amount'] = data['sales_amount'] * (1 + 0.001 * np.arange(n_records))  # Trend
        data['sales_amount'] = data['sales_amount'] * (1 + 0.2 * np.sin(2 * np.pi * np.arange(n_records) / 30))  # Seasonality
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        mask = np.random.random(n_records) < 0.05  # 5% missing values
        df.loc[mask, 'customer_rating'] = np.nan
        
        return df
    
    def _generate_financial_data(self):
        """Generate sample financial data"""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        n_records = len(dates)
        
        # Base values with growth
        time_trend = np.arange(n_records) * 0.02
        revenue = np.random.normal(500000, 50000, n_records) * (1 + time_trend)
        expenses = np.random.normal(350000, 30000, n_records) * (1 + time_trend * 0.8)
        
        data = {
            'date': dates,
            'revenue': revenue.round(2),
            'expenses': expenses.round(2),
            'profit': (revenue - expenses).round(2),
            'assets': np.cumsum(np.random.normal(50000, 5000, n_records)).round(2),
            'liabilities': np.cumsum(np.random.normal(20000, 2000, n_records)).round(2),
            'equity': np.cumsum(np.random.normal(30000, 3000, n_records)).round(2),
            'profit_margin': ((revenue - expenses) / revenue * 100).round(2)
        }
        
        df = pd.DataFrame(data)
        return df
    
    def _generate_customer_data(self):
        """Generate sample customer data"""
        n_customers = 500
        
        data = {
            'customer_id': range(1001, 1001 + n_customers),
            'age': np.random.randint(18, 70, n_customers),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04]),
            'income': np.random.normal(75000, 25000, n_customers).round(2),
            'spending_score': np.random.randint(1, 100, n_customers),
            'loyalty_years': np.random.exponential(3, n_customers).round(1),
            'region': np.random.choice(['Urban', 'Suburban', 'Rural'], n_customers),
            'segment': np.random.choice(['Premium', 'Gold', 'Silver', 'Bronze'], n_customers, p=[0.1, 0.2, 0.3, 0.4]),
            'satisfaction_score': np.random.uniform(1, 5, n_customers).round(1)
        }
        
        # Create correlation between income and spending score
        data['spending_score'] = (data['spending_score'] + (data['income'] - 75000) / 5000).clip(1, 100)
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        mask = np.random.random(n_customers) < 0.03
        df.loc[mask, 'satisfaction_score'] = np.nan
        
        return df
    
    def _generate_timeseries_data(self):
        """Generate sample time series data"""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_records = len(dates)
        
        # Create complex time series with trend, seasonality, and noise
        trend = np.linspace(100, 250, n_records)
        seasonality = 20 * np.sin(2 * np.pi * np.arange(n_records) / 365)  # Yearly seasonality
        weekly_seasonality = 5 * np.sin(2 * np.pi * np.arange(n_records) / 7)  # Weekly seasonality
        noise = np.random.normal(0, 8, n_records)
        
        data = {
            'date': dates,
            'value': trend + seasonality + weekly_seasonality + noise,
            'category': np.random.choice(['A', 'B', 'C'], n_records),
            'sub_category': np.random.choice(['X', 'Y', 'Z'], n_records),
            'metric_1': np.random.normal(50, 10, n_records),
            'metric_2': np.random.normal(100, 20, n_records)
        }
        
        # Add some outliers
        outlier_mask = np.random.random(n_records) < 0.01  # 1% outliers
        data['value'][outlier_mask] = data['value'][outlier_mask] * 3
        
        df = pd.DataFrame(data)
        return df
    
    def clean_data(self, df):
        """Perform basic data cleaning operations"""
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all')
        df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
        
        # Convert date columns if possible
        for col in df_clean.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                except:
                    pass  # Keep as is if conversion fails
        
        # Clean column names
        df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
        
        return df_clean
    
    def validate_data(self, df):
        """Validate data quality and return report"""
        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
        
        return validation_report