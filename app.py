import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os
import tempfile
import time
import json
from datetime import datetime

# For Excel files
import openpyxl

# For handling databases
import sqlite3

# For NLP analysis
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# UI components
import streamlit_option_menu as st_option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from streamlit_lottie import st_lottie
from streamlit_extras.metric_cards import style_metric_cards

# Set page configuration
st.set_page_config(
    page_title="AI Data Analyzer & Dashboard Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'data_overview' not in st.session_state:
    st.session_state.data_overview = None
if 'recommended_visualizations' not in st.session_state:
    st.session_state.recommended_visualizations = []
if 'dashboard_generated' not in st.session_state:
    st.session_state.dashboard_generated = False
if 'data_description' not in st.session_state:
    st.session_state.data_description = ""
if 'filters' not in st.session_state:
    st.session_state.filters = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Function to download NLTK resources if not already present
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Download required NLTK resources
download_nltk_resources()

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
        }
        .stAlert > div {
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
        }
        .visualization-card {
            border: 1px solid #e6e6e6;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: #ffffff;
        }
        .metric-card {
            border: 1px solid #e6e6e6;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: #f8f9fa;
        }
        .filter-card {
            background-color: #f1f3f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .dashboard-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .insight-text {
            font-style: italic;
            color: #6c757d;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Load animations
@st.cache_resource
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

# Helper functions
def get_file_extension(file_name):
    return os.path.splitext(file_name)[1].lower()

def read_data_file(uploaded_file):
    """Read different types of data files into a pandas DataFrame"""
    
    file_name = uploaded_file.name
    st.session_state.file_name = file_name
    file_extension = get_file_extension(file_name)
    
    try:
        if file_extension == '.csv':
            # Try different encodings and delimiters
            try:
                df = pd.read_csv(uploaded_file)
            except:
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                except:
                    try:
                        df = pd.read_csv(uploaded_file, delimiter=';')
                    except:
                        df = pd.read_csv(StringIO(uploaded_file.getvalue().decode('utf-8')), delimiter=None, engine='python')
        
        elif file_extension in ['.xls', '.xlsx']:
            # Get all sheet names
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            
            if len(sheet_names) == 1:
                # If there's only one sheet, read it directly
                df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
            else:
                # Let the user select which sheet to analyze or analyze all
                sheet_option = st.radio(
                    "Multiple sheets detected. How would you like to proceed?",
                    ["Analyze all sheets combined", "Select a specific sheet"],
                    key="sheet_option"
                )
                
                if sheet_option == "Analyze all sheets combined":
                    # Read all sheets and combine them
                    all_dfs = []
                    for sheet in sheet_names:
                        sheet_df = pd.read_excel(uploaded_file, sheet_name=sheet)
                        # Add a column to identify the sheet source
                        sheet_df['_sheet_name'] = sheet
                        all_dfs.append(sheet_df)
                    
                    # Combine all sheets
                    df = pd.concat(all_dfs, ignore_index=True)
                    st.info(f"Analyzing {len(sheet_names)} sheets combined into one dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
                else:
                    # Let the user select a specific sheet
                    selected_sheet = st.selectbox("Select a sheet to analyze:", sheet_names, key="sheet_select")
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    st.info(f"Analyzing sheet: {selected_sheet}")
        
        elif file_extension == '.json':
            df = pd.read_json(uploaded_file)
        
        elif file_extension in ['.db', '.sqlite']:
            # For SQLite database files, we need to save it temporarily and then read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Connect to the SQLite database
            conn = sqlite3.connect(tmp_path)
            
            # Get all table names
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if not tables:
                st.error("No tables found in the database.")
                return None
            
            # Create a dictionary of all tables
            table_data = {}
            for table in tables:
                table_name = table[0]
                table_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                table_data[table_name] = table_df
            
            # Close the connection and remove the temporary file
            conn.close()
            os.unlink(tmp_path)
            
            # If there's only one table, return it directly
            if len(table_data) == 1:
                df = list(table_data.values())[0]
            else:
                # Otherwise, let the user select a table
                selected_table = st.selectbox("Select a table from the database:", list(table_data.keys()))
                df = table_data[selected_table]
        
        elif file_extension == '.txt':
            # Try to parse as CSV first
            try:
                df = pd.read_csv(uploaded_file, delimiter=None, engine='python')
            except:
                # If that fails, read as plain text and try to structure it
                content = uploaded_file.getvalue().decode('utf-8')
                lines = content.strip().split('\n')
                
                # Check if each line has the same number of fields when split by common delimiters
                for delimiter in [',', '\t', '|', ';']:
                    fields_per_line = [len(line.split(delimiter)) for line in lines]
                    if len(set(fields_per_line)) == 1 and fields_per_line[0] > 1:
                        # All lines have the same number of fields, try to parse as CSV
                        df = pd.read_csv(StringIO(content), delimiter=delimiter)
                        break
                else:
                    # If no consistent delimiter is found, just create a DataFrame with a 'text' column
                    df = pd.DataFrame({'text': lines})
        
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
        return None

def analyze_data(df):
    """Analyze the data and return insights"""
    if df is None or df.empty:
        return None
    
    num_rows, num_cols = df.shape
    
    # Basic information
    analysis = {
        "num_rows": num_rows,
        "num_columns": num_cols,
        "column_names": list(df.columns),
        "column_types": {col: str(df[col].dtype) for col in df.columns},
        "null_counts": df.isnull().sum().to_dict(),
        "unique_counts": {col: df[col].nunique() for col in df.columns},
        "column_descriptions": {},
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "text_columns": [],
        "binary_columns": []
    }
    
    # Sample data
    analysis["sample_data"] = df.head(5).to_dict('records')
    
    # Categorize columns
    for col in df.columns:
        col_type = df[col].dtype
        unique_ratio = analysis["unique_counts"][col] / num_rows if num_rows > 0 else 0
        
        # Identify datetime columns
        if col_type == 'datetime64[ns]' or (df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any() and df[col].nunique() > 1):
            analysis["datetime_columns"].append(col)
            try:
                df[col] = pd.to_datetime(df[col])
                analysis["column_types"][col] = "datetime"
            except:
                pass
            
        # Identify numeric columns
        elif pd.api.types.is_numeric_dtype(col_type) and unique_ratio > 0.01:
            analysis["numeric_columns"].append(col)
            
            # Calculate basic statistics for numeric columns
            if col not in analysis["column_descriptions"]:
                analysis["column_descriptions"][col] = {}
            
            analysis["column_descriptions"][col].update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None
            })
            
        # Identify binary columns
        elif df[col].nunique() == 2:
            analysis["binary_columns"].append(col)
            
            # Get the two values
            values = df[col].dropna().unique().tolist()
            
            if col not in analysis["column_descriptions"]:
                analysis["column_descriptions"][col] = {}
                
            analysis["column_descriptions"][col].update({
                "values": values,
                "counts": df[col].value_counts().to_dict()
            })
            
        # Identify categorical columns
        elif unique_ratio < 0.5 and df[col].nunique() < 50:
            analysis["categorical_columns"].append(col)
            
            # Get top categories and their counts
            if col not in analysis["column_descriptions"]:
                analysis["column_descriptions"][col] = {}
                
            value_counts = df[col].value_counts().head(10).to_dict()
            analysis["column_descriptions"][col].update({
                "top_values": value_counts
            })
            
        # Identify text columns
        elif df[col].dtype == 'object' and df[col].str.len().mean() > 10:
            analysis["text_columns"].append(col)
            
            if col not in analysis["column_descriptions"]:
                analysis["column_descriptions"][col] = {}
                
            # Get average text length and sample texts
            analysis["column_descriptions"][col].update({
                "avg_length": float(df[col].str.len().mean()) if not pd.isna(df[col].str.len().mean()) else None,
                "sample_texts": df[col].dropna().sample(min(3, df[col].count())).tolist() if df[col].count() > 0 else []
            })
    
    # Detect potential key relationships
    analysis["potential_relationships"] = []
    
    # Look for date patterns and time series
    if analysis["datetime_columns"]:
        for date_col in analysis["datetime_columns"]:
            # Check if the date column has a good distribution (not all same date)
            if df[date_col].nunique() > 5:
                # Find numeric columns that might be time series with this date
                for num_col in analysis["numeric_columns"]:
                    analysis["potential_relationships"].append({
                        "type": "time_series",
                        "x_column": date_col,
                        "y_column": num_col,
                        "description": f"Time series of {num_col} over {date_col}"
                    })
    
    # Look for categorical vs. numeric relationships (potential group analysis)
    for cat_col in analysis["categorical_columns"] + analysis["binary_columns"]:
        for num_col in analysis["numeric_columns"]:
            analysis["potential_relationships"].append({
                "type": "categorical_vs_numeric",
                "category_column": cat_col,
                "numeric_column": num_col,
                "description": f"Distribution of {num_col} across different {cat_col} categories"
            })
    
    # Look for correlations between numeric columns
    if len(analysis["numeric_columns"]) > 1:
        try:
            corr_matrix = df[analysis["numeric_columns"]].corr()
            
            # Find strong correlations (absolute value > 0.5)
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j and abs(corr_matrix.iloc[i, j]) > 0.5:
                        analysis["potential_relationships"].append({
                            "type": "correlation",
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_matrix.iloc[i, j]),
                            "description": f"Strong {'positive' if corr_matrix.iloc[i, j] > 0 else 'negative'} correlation between {col1} and {col2}"
                        })
        except:
            pass  # Skip correlation analysis if it fails
    
    # Generate recommended visualizations based on the analysis
    visualizations = []
    
    # For numeric columns: histograms and box plots
    for col in analysis["numeric_columns"][:5]:  # Limit to top 5
        visualizations.append({
            "type": "histogram",
            "column": col,
            "title": f"Distribution of {col}",
            "description": f"Shows the distribution of values for {col}"
        })
        
        visualizations.append({
            "type": "box_plot",
            "column": col,
            "title": f"Box Plot of {col}",
            "description": f"Shows the quartiles, median, and outliers for {col}"
        })
    
    # For categorical columns: bar charts
    for col in analysis["categorical_columns"][:5]:  # Limit to top 5
        visualizations.append({
            "type": "bar_chart",
            "column": col,
            "title": f"Count of {col} Categories",
            "description": f"Shows the frequency of each category in {col}"
        })
    
    # For time series: line charts
    for relation in [r for r in analysis["potential_relationships"] if r["type"] == "time_series"][:5]:
        visualizations.append({
            "type": "line_chart",
            "x_column": relation["x_column"],
            "y_column": relation["y_column"],
            "title": f"{relation['y_column']} over {relation['x_column']}",
            "description": relation["description"]
        })
    
    # For correlations: scatter plots
    for relation in [r for r in analysis["potential_relationships"] if r["type"] == "correlation"][:5]:
        visualizations.append({
            "type": "scatter_plot",
            "x_column": relation["column1"],
            "y_column": relation["column2"],
            "title": f"Correlation between {relation['column1']} and {relation['column2']}",
            "description": relation["description"]
        })
    
    # For categorical vs numeric: box plots or violin plots
    for relation in [r for r in analysis["potential_relationships"] if r["type"] == "categorical_vs_numeric"][:5]:
        # Use boxplot only if categories are not too many
        if df[relation["category_column"]].nunique() <= 10:
            visualizations.append({
                "type": "grouped_box_plot",
                "category_column": relation["category_column"],
                "numeric_column": relation["numeric_column"],
                "title": f"{relation['numeric_column']} by {relation['category_column']}",
                "description": relation["description"]
            })
    
    # Add correlation heatmap if there are enough numeric columns
    if len(analysis["numeric_columns"]) > 2:
        visualizations.append({
            "type": "correlation_heatmap",
            "columns": analysis["numeric_columns"],
            "title": "Correlation Heatmap",
            "description": "Shows the correlation between numeric variables"
        })
    
    # Generate suitable filters
    filters = []
    
    # Add date range filters
    for col in analysis["datetime_columns"]:
        filters.append({
            "type": "date_range",
            "column": col,
            "title": f"Filter by {col}"
        })
    
    # Add categorical filters
    for col in analysis["categorical_columns"] + analysis["binary_columns"]:
        if df[col].nunique() <= 15:  # Only add filter if there aren't too many categories
            filters.append({
                "type": "categorical",
                "column": col,
                "title": f"Filter by {col}",
                "values": df[col].dropna().unique().tolist()
            })
    
    # Add numeric range filters
    for col in analysis["numeric_columns"]:
        filters.append({
            "type": "numeric_range",
            "column": col,
            "title": f"Filter by {col}",
            "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0,
            "max": float(df[col].max()) if not pd.isna(df[col].max()) else 100
        })
    
    # Generate an overall data description
    data_type = guess_data_domain(df)
    time_dimension = analysis["datetime_columns"][0] if analysis["datetime_columns"] else None
    key_metrics = analysis["numeric_columns"][:3] if analysis["numeric_columns"] else []
    categories = analysis["categorical_columns"][:3] if analysis["categorical_columns"] else []
    
    description = f"This dataset contains {num_rows} records with {num_cols} columns. "
    description += f"It appears to be {data_type} data. "
    
    if time_dimension:
        description += f"The data has a time dimension ({time_dimension}). "
    
    if key_metrics:
        description += f"Key metrics include {', '.join(key_metrics)}. "
    
    if categories:
        description += f"Main categories include {', '.join(categories)}. "
    
    # Add data overview
    analysis["data_description"] = description
    analysis["recommended_visualizations"] = visualizations
    analysis["recommended_filters"] = filters
    analysis["data_domain"] = data_type
    
    return analysis

def guess_data_domain(df):
    """Guess the domain of the data based on column names and content"""
    # Convert column names to strings before lowercasing to handle non-string column names
    column_names = [str(col).lower() for col in df.columns]
    
    # Sales/E-commerce data
    sales_terms = ['sales', 'revenue', 'product', 'customer', 'order', 'price', 'discount', 'quantity']
    if any(term in ' '.join(column_names) for term in sales_terms):
        return "sales/e-commerce"
    
    # Financial data
    finance_terms = ['profit', 'loss', 'expense', 'income', 'budget', 'cost', 'transaction', 'account', 'balance']
    if any(term in ' '.join(column_names) for term in finance_terms):
        return "financial"
    
    # Healthcare data
    health_terms = ['patient', 'diagnosis', 'treatment', 'doctor', 'hospital', 'medical', 'disease', 'health']
    if any(term in ' '.join(column_names) for term in health_terms):
        return "healthcare"
    
    # HR/Employee data
    hr_terms = ['employee', 'salary', 'department', 'hire', 'position', 'hr', 'performance', 'rating']
    if any(term in ' '.join(column_names) for term in hr_terms):
        return "human resources"
    
    # Marketing data
    marketing_terms = ['campaign', 'click', 'conversion', 'ad', 'marketing', 'channel', 'lead', 'engagement']
    if any(term in ' '.join(column_names) for term in marketing_terms):
        return "marketing"
    
    # Education data
    education_terms = ['student', 'course', 'grade', 'teacher', 'school', 'class', 'education', 'score', 'test']
    if any(term in ' '.join(column_names) for term in education_terms):
        return "education"
    
    # Default
    return "general business"

def render_visualization(df, viz_config, container):
    """Render a visualization based on the configuration"""
    viz_type = viz_config["type"]
    title = viz_config.get("title", "")
    
    try:
        if viz_type == "histogram":
            column = viz_config["column"]
            fig = px.histogram(df, x=column, title=title)
            container.plotly_chart(fig, use_container_width=True)
            
            # Add descriptive statistics
            stats = df[column].describe()
            container.markdown(f"""
            **Key statistics:**
            - Mean: {stats['mean']:.2f}
            - Median: {stats['50%']:.2f}
            - Standard Deviation: {stats['std']:.2f}
            - Min: {stats['min']:.2f}
            - Max: {stats['max']:.2f}
            """)
    
    except Exception as e:
        container.error(f"Error rendering visualization: {str(e)}")
        return False
    
    return True

def apply_filters(df, filters):
    """Apply selected filters to the dataframe"""
    if not filters:
        return df
    
    filtered_df = df.copy()
    filter_description = []
    
    for filter_config in filters:
        filter_type = filter_config["type"]
        
        if filter_type == "date_range" and filter_config.get("selected", False):
            column = filter_config["column"]
            start_date = filter_config.get("start_date")
            end_date = filter_config.get("end_date")
            
            if start_date and end_date:
                filtered_df = filtered_df[(filtered_df[column] >= start_date) & (filtered_df[column] <= end_date)]
                filter_description.append(f"{column} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        
        elif filter_type == "categorical" and filter_config.get("selected", False):
            column = filter_config["column"]
            selected_values = filter_config.get("selected_values", [])
            
            if selected_values:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
                filter_description.append(f"{column} is {', '.join(map(str, selected_values))}")
        
        elif filter_type == "numeric_range" and filter_config.get("selected", False):
            column = filter_config["column"]
            min_value = filter_config.get("min_value")
            max_value = filter_config.get("max_value")
            
            if min_value is not None and max_value is not None:
                filtered_df = filtered_df[(filtered_df[column] >= min_value) & (filtered_df[column] <= max_value)]
                filter_description.append(f"{column} between {min_value} and {max_value}")
    
    return filtered_df, filter_description

def generate_dashboard(df, data_overview, selected_visualizations, filters):
    """Generate a dashboard with the selected visualizations and filters"""
    st.title("📊 AI-Generated Dashboard")
    
    # Display data information
    with st.expander("About this dataset", expanded=False):
        st.markdown(f"**File name:** {st.session_state.file_name}")
        st.markdown(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
        st.markdown(f"**Data Description:** {data_overview['data_description']}")
        
        st.subheader("Sample Data")
        st.dataframe(df.head(5))
    
    # Apply filters
    filter_container = st.container()
    with filter_container:
        st.subheader("Filters")
        
        filter_cols = st.columns(min(4, len(filters)))
        updated_filters = []
        
        for i, filter_config in enumerate(filters):
            filter_type = filter_config["type"]
            column = filter_config["column"]
            col_idx = i % len(filter_cols)
            
            with filter_cols[col_idx]:
                st.markdown(f"**{filter_config['title']}**")
                updated_filter = filter_config.copy()
                
                if filter_type == "date_range":
                   try:
                        min_date = df[column].min()
                        max_date = df[column].max()
                        
                        start_date = st.date_input(
                            f"Start date",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key=f"start_{column}"
                        )
                        
                        end_date = st.date_input(
                            f"End date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key=f"end_{column}"
                        )
                        
                        updated_filter["start_date"] = pd.Timestamp(start_date)
                        updated_filter["end_date"] = pd.Timestamp(end_date)
                        updated_filter["selected"] = True
                except Exception as e:
                        st.error(f"Error with date filter for {column}: {str(e)}")
                        updated_filter["selected"] = False
                        
    # Display visualizations in a grid
    st.subheader("Visualizations")
    
    num_cols = 2  # Number of columns in the grid
    viz_rows = [selected_visualizations[i:i+num_cols] for i in range(0, len(selected_visualizations), num_cols)]
    
    for row in viz_rows:
        cols = st.columns(num_cols)
        
        for i, viz in enumerate(row):
            with cols[i]:
                st.markdown(f"**{viz['title']}**")
                render_visualization(filtered_df, viz, cols[i])
    
    return True

def main():
    st.title("🧠 AI Data Analyzer & Dashboard Generator")
    st.write("Upload your data file and let AI analyze it and create a custom dashboard for you.")
    
    with st.sidebar:
        st.image("https://img.icons8.com/pulsar-color/96/data-configuration.png", width=80)
        st.header("Data Analysis Options")
        
        uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls', 'json', 'txt', 'db', 'sqlite'])
        
        if uploaded_file is not None:
            if st.button("Analyze Data", type="primary"):
                with st.spinner("Reading and analyzing data..."):
                    # Reset session state
                    st.session_state.analysis_complete = False
                    st.session_state.dashboard_generated = False
                    
                    # Read the data file
                    df = read_data_file(uploaded_file)
                    
                    if df is not None and not df.empty:
                        # Store the data
                        st.session_state.data = df
                        
                        # Analyze the data
                        st.session_state.data_overview = analyze_data(df)
                        
                        if st.session_state.data_overview:
                            # Get recommended visualizations
                            st.session_state.recommended_visualizations = st.session_state.data_overview.get("recommended_visualizations", [])
                            
                            # Get recommended filters
                            st.session_state.filters = st.session_state.data_overview.get("recommended_filters", [])
                            
                            # Set data description
                            st.session_state.data_description = st.session_state.data_overview.get("data_description", "")
                            
                            # Mark analysis as complete
                            st.session_state.analysis_complete = True
                            
                            # Store a copy of the data (for filters)
                            st.session_state.processed_data = df.copy()
                            
                            # Rerun the app to update the UI
                            st.rerun()
                    else:
                        st.error("Failed to read or analyze the data. Please check the file format.")
        
        if st.session_state.analysis_complete:
            st.success("Analysis complete!")
            
            # Show data description
            st.subheader("Data Overview")
            st.write(st.session_state.data_description)
            
            # Select visualizations
            st.subheader("Select Visualizations")
            
            selected_viz_indices = []
            for i, viz in enumerate(st.session_state.recommended_visualizations):
                selected = st.checkbox(viz["title"], value=True, key=f"viz_{i}")
                if selected:
                    selected_viz_indices.append(i)
            
            selected_visualizations = [st.session_state.recommended_visualizations[i] for i in selected_viz_indices]
            
            # Generate dashboard button
            if st.button("Generate Dashboard", type="primary"):
                st.session_state.dashboard_generated = True
                st.rerun()
    
    # Display dashboard content in the main area
    if st.session_state.dashboard_generated and st.session_state.data is not None:
        generate_dashboard(
            st.session_state.processed_data,
            st.session_state.data_overview,
            [st.session_state.recommended_visualizations[i] for i in range(len(st.session_state.recommended_visualizations)) if st.checkbox(f"viz_{i}", value=True, key=f"viz_dashboard_{i}")],
            st.session_state.filters
        )
    else:
        # Display instructions/welcome message
        st.markdown("""
        ## 👋 Welcome to AI Data Analyzer & Dashboard Generator!
        
        This application helps you instantly analyze your data and create beautiful dashboards with just a few clicks.
        
        ### How It Works:
        1. **Upload your data file** (CSV, Excel, SQLite, etc.)
        2. **Click "Analyze Data"** to let AI examine your dataset
        3. **Select visualizations** you'd like to include
        4. **Generate your dashboard** with a single click
        
        ### Features:
        - Automatic data analysis and insights
        - AI-recommended visualizations based on your data
        - Interactive filters to explore your data
        - Key metrics and trends detection
        - Easy-to-understand explanations of findings
        
        Upload your file in the sidebar to get started!
        """)

if __name__ == "__main__":
    main()
                        
                        end_date = st.date_input(
                            f"End date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key=f"end_{column}"
                        )
                        
                        updated_filter["start_date"] = pd.Timestamp(start_date)
                        updated_filter["end_date"] = pd.Timestamp(end_date)
                        updated_filter["selected"] = True
                    except:
                        st.error(f"Error with date filter for {column}")
                        updated_filter["selected"] = False
                
                elif filter_type == "categorical":
                    try:
                        all_values = filter_config.get("values", [])
                        default_all = st.checkbox("Select all", key=f"all_{column}")
                        
                        if default_all:
                            selected_values = st.multiselect(
                                f"Values",
                                options=all_values,
                                default=all_values,
                                key=f"multi_{column}"
                            )
                        else:
                            selected_values = st.multiselect(
                                f"Values",
                                options=all_values,
                                key=f"multi_{column}"
                            )
                        
                        updated_filter["selected_values"] = selected_values
                        updated_filter["selected"] = len(selected_values) > 0
                    except:
                        st.error(f"Error with categorical filter for {column}")
                        updated_filter["selected"] = False
                
                elif filter_type == "numeric_range":
                    try:
                        min_val = filter_config.get("min", float(df[column].min()))
                        max_val = filter_config.get("max", float(df[column].max()))
                        
                        min_value, max_value = st.slider(
                            f"Range",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"range_{column}"
                        )
                        
                        updated_filter["min_value"] = min_value
                        updated_filter["max_value"] = max_value
                        updated_filter["selected"] = min_value > min_val or max_value < max_val
                    except:
                        st.error(f"Error with numeric filter for {column}")
                        updated_filter["selected"] = False
            
            updated_filters.append(updated_filter)
    
    # Apply filters to dataframe
    filtered_df, filter_description = apply_filters(df, updated_filters)
    
    # Show filter summary
    if filter_description:
        st.info(f"Filtered data: {', '.join(filter_description)} (Showing {filtered_df.shape[0]} of {df.shape[0]} rows)")
    
    # Dashboard metrics (summary stats)
    metric_container = st.container()
    with metric_container:
        st.subheader("Key Metrics")
        
        # Find numeric columns to use as metrics
        numeric_cols = data_overview["numeric_columns"]
        datetime_cols = data_overview["datetime_columns"]
        categorical_cols = data_overview["categorical_columns"]
        
        if numeric_cols:
            metric_cols = st.columns(min(4, len(numeric_cols)))
            
            for i, col in enumerate(numeric_cols[:4]):  # Show up to 4 metrics
                with metric_cols[i]:
                    current_value = filtered_df[col].mean()
                    
                    # Calculate delta if we have a time dimension
                    delta = None
                    if datetime_cols and filtered_df.shape[0] > 0:
                        time_col = datetime_cols[0]
                        try:
                            filtered_df['temp_period'] = pd.to_datetime(filtered_df[time_col]).dt.to_period('M')
                            
                            # Get the mean value for the most recent and previous period
                            period_values = filtered_df.groupby('temp_period')[col].mean()
                            
                            if len(period_values) >= 2:
                                current_period = period_values.iloc[-1]
                                previous_period = period_values.iloc[-2]
                                delta = ((current_period - previous_period) / previous_period) * 100
                        except:
                            pass
                    
                    st.metric(
                        label=col,
                        value=f"{current_value:.2f}",
                        delta=f"{delta:.1f}%" if delta is not None else None
                    )
            
        elif viz_type == "box_plot":
            column = viz_config["column"]
            fig = px.box(df, y=column, title=title)
            container.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "bar_chart":
            column = viz_config["column"]
            value_counts = df[column].value_counts().reset_index()
            value_counts.columns = [column, 'count']
            
            # Sort by count descending and take top 20 if there are more
            value_counts = value_counts.sort_values('count', ascending=False).head(20)
            
            fig = px.bar(value_counts, x=column, y='count', title=title)
            container.plotly_chart(fig, use_container_width=True)
            
            # Add insights
            top_category = value_counts.iloc[0][column]
            top_percentage = (value_counts.iloc[0]['count'] / df.shape[0]) * 100
            container.markdown(f"""
            **Insights:**
            - Most frequent category: {top_category} ({top_percentage:.1f}% of data)
            - Number of unique categories: {df[column].nunique()}
            """)
            
        elif viz_type == "line_chart":
            x_column = viz_config["x_column"]
            y_column = viz_config["y_column"]
            
            # Make sure x_column is datetime
            if df[x_column].dtype != 'datetime64[ns]':
                df[x_column] = pd.to_datetime(df[x_column])
            
            # Resample by day, week, month, or year depending on the date range
            date_range = (df[x_column].max() - df[x_column].min()).days
            
            if date_range > 365 * 3:  # More than 3 years
                time_group = 'M'
                time_format = '%Y-%m'
            elif date_range > 365:  # More than 1 year
                time_group = 'W'
                time_format = '%Y-%W'
            else:
                time_group = 'D'
                time_format = '%Y-%m-%d'
            
            # Group by time period and calculate mean
            df_grouped = df.groupby(df[x_column].dt.strftime(time_format)).agg({y_column: 'mean'}).reset_index()
            
            fig = px.line(df_grouped, x=x_column, y=y_column, title=title)
            container.plotly_chart(fig, use_container_width=True)
            
            # Add trend analysis
            if len(df_grouped) > 2:
                try:
                    first_value = df_grouped[y_column].iloc[0]
                    last_value = df_grouped[y_column].iloc[-1]
                    pct_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                    
                    trend = "increasing" if pct_change > 0 else "decreasing"
                    container.markdown(f"""
                    **Trend Analysis:**
                    - Overall trend: {trend}
                    - Change from start to end: {pct_change:.1f}%
                    - Average value: {df_grouped[y_column].mean():.2f}
                    """)
                except:
                    pass
            
        elif viz_type == "scatter_plot":
            x_column = viz_config["x_column"]
            y_column = viz_config["y_column"]
            
            fig = px.scatter(df, x=x_column, y=y_column, title=title)
            
            # Add trendline
            fig.update_layout(
                shapes=[
                    dict(
                        type='line',
                        xref='x',
                        yref='y',
                        x0=df[x_column].min(),
                        y0=df[y_column].min(),
                        x1=df[x_column].max(),
                        y1=df[y_column].max(),
                        line=dict(
                            color="red",
                            width=2,
                            dash="dot",
                        )
                    )
                ]
            )
            
            container.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation
            correlation = df[[x_column, y_column]].corr().iloc[0, 1]
            correlation_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
            correlation_direction = "positive" if correlation > 0 else "negative"
            
            container.markdown(f"""
            **Correlation Analysis:**
            - Correlation coefficient: {correlation:.3f}
            - This indicates a {correlation_strength} {correlation_direction} relationship
            """)
            
        elif viz_type == "grouped_box_plot":
            category_column = viz_config["category_column"]
            numeric_column = viz_config["numeric_column"]
            
            # Limit to top 10 categories if there are more
            if df[category_column].nunique() > 10:
                top_categories = df[category_column].value_counts().nlargest(10).index.tolist()
                filtered_df = df[df[category_column].isin(top_categories)]
            else:
                filtered_df = df
            
            fig = px.box(filtered_df, x=category_column, y=numeric_column, title=title)
            container.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display statistics by group
            stats_by_group = df.groupby(category_column)[numeric_column].agg(['mean', 'median', 'std']).sort_values('mean', ascending=False)
            
            # Format the stats table
            stats_table = pd.DataFrame({
                'Category': stats_by_group.index,
                'Mean': stats_by_group['mean'].round(2),
                'Median': stats_by_group['median'].round(2),
                'Std Dev': stats_by_group['std'].round(2)
            }).reset_index(drop=True)
            
            container.markdown("**Statistics by Category:**")
            container.dataframe(stats_table)
            
        elif viz_type == "correlation_heatmap":
            columns = viz_config["columns"]
            
            # Limit to a reasonable number of columns
            if len(columns) > 15:
                columns = columns[:15]
            
            # Calculate correlation matrix
            corr_matrix = df[columns].corr().round(2)
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title=title
            )
            
            container.plotly_chart(fig, use_container_width=True)
            
            # Identify strongest correlations
            corr_pairs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Only consider upper triangle of correlation matrix
                        corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))
            
            # Sort by absolute correlation and get top 5
            top_corrs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
            
            if top_corrs:
                container.markdown("**Strongest Correlations:**")
                for col1, col2, corr in top_corrs:
                    direction = "positive" if corr > 0 else "negative"
                    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                    container.markdown(f"- {col1} and {col2}: {corr:.2f} ({strength} {direction})")
        
        elif viz_type == "pie_chart":
            column = viz_config["column"]
            
            # Get value counts and limit to top 10 categories + "Other"
            value_counts = df[column].value_counts()
            
            if len(value_counts) > 10:
                top_values = value_counts.nlargest(10)
                other_value = pd.Series([value_counts.iloc[10:].sum()], index=["Other"])
                pie_data = pd.concat([top_values, other_value])
            else:
                pie_data = value_counts
            
            fig = px.pie(
                values=pie_data.values,
                names=pie_data.index,
                title=title
            )
            
            container.plotly_chart(fig, use_container_width=True)
            
            # Add insights
            top_category = pie_data.index[0]
            top_percentage = (pie_data.iloc[0] / pie_data.sum()) * 100
            container.markdown(f"""
            **Insights:**
            - Largest category: {top_category} ({top_percentage:.1f}% of total)
            - Number of categories: {df[column].nunique()}
            """)
