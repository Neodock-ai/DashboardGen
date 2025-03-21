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
import requests
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

# Set page configuration
st.set_page_config(
    page_title="AI-Enhanced Data Analyzer & Dashboard Generator",
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
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
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
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'using_openai' not in st.session_state:
    st.session_state.using_openai = False
if 'using_local_analysis' not in st.session_state:
    st.session_state.using_local_analysis = True
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = "gpt-3.5-turbo"
if 'selected_viz_indices' not in st.session_state:
    st.session_state.selected_viz_indices = []

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
        .ai-insight-card {
            background-color: #f0f7ff;
            border-left: 4px solid #0066cc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .ai-recommendation {
            background-color: #f7f7f7;
            border: 1px solid #e0e0e0;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

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
        
def render_visualization(df, viz_config, container):
    """Render a visualization based on the configuration"""
    viz_type = viz_config["type"]
    title = viz_config.get("title", "")
    
    try:
        if viz_type == "histogram":
            column = viz_config["column"]
            
            # Check if column exists and has valid data
            if column in df.columns and df[column].count() > 0:
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
            else:
                container.warning(f"Cannot create histogram for {column}. Column may be missing or contain no valid data.")
        
        elif viz_type == "box_plot":
            column = viz_config["column"]
            
            # Check if column exists and has valid data
            if column in df.columns and df[column].count() > 0:
                fig = px.box(df, y=column, title=title)
                container.plotly_chart(fig, use_container_width=True)
            else:
                container.warning(f"Cannot create box plot for {column}. Column may be missing or contain no valid data.")
            
        elif viz_type == "bar_chart":
            column = viz_config["column"]
            
            # Check if column exists and has valid data
            if column in df.columns and df[column].count() > 0:
                value_counts = df[column].value_counts().reset_index()
                value_counts.columns = [column, 'count']
                
                # Sort by count descending and take top 20 if there are more
                value_counts = value_counts.sort_values('count', ascending=False).head(20)
                
                if not value_counts.empty:
                    fig = px.bar(value_counts, x=column, y='count', title=title)
                    container.plotly_chart(fig, use_container_width=True)
                    
                    # Add insights
                    top_category = value_counts.iloc[0][column] if len(value_counts) > 0 else "N/A"
                    top_percentage = (value_counts.iloc[0]['count'] / df.shape[0]) * 100 if len(value_counts) > 0 else 0
                    container.markdown(f"""
                    **Insights:**
                    - Most frequent category: {top_category} ({top_percentage:.1f}% of data)
                    - Number of unique categories: {df[column].nunique()}
                    """)
                else:
                    container.warning(f"Not enough data to create bar chart for {column}.")
            else:
                container.warning(f"Cannot create bar chart for {column}. Column may be missing or contain no valid data.")
            
        elif viz_type == "line_chart":
            x_column = viz_config["x_column"]
            y_column = viz_config["y_column"]
            
            # Check if columns exist and have valid data
            if x_column in df.columns and y_column in df.columns and df[[x_column, y_column]].dropna().shape[0] > 1:
                # Try to convert x_column to datetime if it's not already
                if df[x_column].dtype != 'datetime64[ns]':
                    try:
                        df[x_column] = pd.to_datetime(df[x_column])
                    except:
                        pass
                
                # Special handling for datetime x-axis
                if pd.api.types.is_datetime64_dtype(df[x_column]):
                    # Resample by day, week, month, or year depending on the date range
                    try:
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
                    except Exception as e:
                        container.error(f"Error creating time series: {str(e)}")
                else:
                    # Regular line plot for non-datetime x-axis
                    fig = px.line(df, x=x_column, y=y_column, title=title)
                    container.plotly_chart(fig, use_container_width=True)
            else:
                container.warning(f"Cannot create line chart for {x_column} vs {y_column}. Columns may be missing or contain insufficient valid data.")
            
        elif viz_type == "scatter_plot":
            x_column = viz_config["x_column"]
            y_column = viz_config["y_column"]
            
            # Check if columns exist and have valid data
            if x_column in df.columns and y_column in df.columns and df[[x_column, y_column]].dropna().shape[0] > 1:
                fig = px.scatter(df, x=x_column, y=y_column, title=title)
                
                # Add trendline if there's enough data
                if df[[x_column, y_column]].dropna().shape[0] > 2:
                    try:
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
                        
                        # Calculate correlation
                        correlation = df[[x_column, y_column]].corr().iloc[0, 1]
                        correlation_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
                        correlation_direction = "positive" if correlation > 0 else "negative"
                        
                        container.plotly_chart(fig, use_container_width=True)
                        container.markdown(f"""
                        **Correlation Analysis:**
                        - Correlation coefficient: {correlation:.3f}
                        - This indicates a {correlation_strength} {correlation_direction} relationship
                        """)
                    except:
                        # Simpler display if correlation calculation fails
                        container.plotly_chart(fig, use_container_width=True)
                else:
                    container.plotly_chart(fig, use_container_width=True)
            else:
                container.warning(f"Cannot create scatter plot for {x_column} vs {y_column}. Columns may be missing or contain insufficient valid data.")
            
        elif viz_type == "grouped_box_plot":
            category_column = viz_config["category_column"]
            numeric_column = viz_config["numeric_column"]
            
            # Check if columns exist and have valid data
            if (category_column in df.columns and numeric_column in df.columns and 
                df[category_column].nunique() > 0 and df[numeric_column].count() > 0):
                
                # Check if dataframe is not empty
                if df.shape[0] > 0 and df[category_column].nunique() > 0:
                    # Limit to top 10 categories if there are more
                    if df[category_column].nunique() > 10:
                        top_categories = df[category_column].value_counts().nlargest(10).index.tolist()
                        filtered_df = df[df[category_column].isin(top_categories)]
                    else:
                        filtered_df = df
                    
                    if filtered_df.shape[0] > 0:  # Make sure we still have data after filtering
                        fig = px.box(filtered_df, x=category_column, y=numeric_column, title=title)
                        container.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate and display statistics by group
                        stats_by_group = df.groupby(category_column)[numeric_column].agg(['mean', 'median', 'std']).sort_values('mean', ascending=False)
                        
                        if not stats_by_group.empty:
                            # Format the stats table
                            stats_table = pd.DataFrame({
                                'Category': stats_by_group.index,
                                'Mean': stats_by_group['mean'].round(2),
                                'Median': stats_by_group['median'].round(2),
                                'Std Dev': stats_by_group['std'].round(2)
                            }).reset_index(drop=True)
                            
                            container.markdown("**Statistics by Category:**")
                            container.dataframe(stats_table)
                        else:
                            container.warning("Not enough data to calculate statistics by group.")
                    else:
                        container.warning("Not enough data after filtering categories.")
                else:
                    container.warning("Not enough data to create grouped box plot.")
            else:
                container.warning(f"Cannot create grouped box plot for {category_column} and {numeric_column}. Columns may be missing or contain insufficient valid data.")
            
        elif viz_type == "correlation_heatmap":
            columns = viz_config["columns"]
            
            # Filter out columns that don't exist in the dataframe
            valid_columns = [col for col in columns if col in df.columns]
            
            # Check if we have at least 2 valid columns
            if len(valid_columns) >= 2:
                # Limit to a reasonable number of columns
                if len(valid_columns) > 15:
                    valid_columns = valid_columns[:15]
                
                # Drop rows with any NaNs in the selected columns
                numeric_df = df[valid_columns].dropna()
                
                if numeric_df.shape[0] > 5:  # Ensure we have enough data
                    # Calculate correlation matrix
                    try:
                        corr_matrix = numeric_df.corr().round(2)
                        
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
                        else:
                            container.info("No strong correlations found between variables.")
                    except Exception as e:
                        container.error(f"Error creating correlation heatmap: {str(e)}")
                else:
                    container.warning("Not enough complete data rows to create a correlation heatmap.")
            else:
                container.warning("Not enough numeric columns available for correlation heatmap.")
        
        elif viz_type == "pie_chart":
            column = viz_config["column"]
            
            # Check if column exists and has valid data
            if column in df.columns and df[column].count() > 0:
                # Get value counts and limit to top 10 categories + "Other"
                value_counts = df[column].value_counts()
                
                if len(value_counts) > 0:
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
                    if not pie_data.empty:
                        top_category = pie_data.index[0] if len(pie_data) > 0 else "N/A"
                        top_percentage = (pie_data.iloc[0] / pie_data.sum()) * 100 if len(pie_data) > 0 else 0
                        container.markdown(f"""
                        **Insights:**
                        - Largest category: {top_category} ({top_percentage:.1f}% of total)
                        - Number of categories: {df[column].nunique()}
                        """)
                    else:
                        container.markdown("**Insights:** No data available for this category.")
                else:
                    container.warning("No data available to create pie chart.")
            else:
                container.warning(f"Cannot create pie chart for {column}. Column may be missing or contain no valid data.")
    
    except Exception as e:
        container.error(f"Error rendering visualization: {str(e)}")
        return False
    
    return True

def apply_filters(df, filters):
    """Apply selected filters to the dataframe"""
    if not filters:
        return df, []
    
    filtered_df = df.copy()
    filter_description = []
    
    for filter_config in filters:
        filter_type = filter_config["type"]
        
        if filter_type == "date_range" and filter_config.get("selected", False):
            column = filter_config["column"]
            start_date = filter_config.get("start_date")
            end_date = filter_config.get("end_date")
            
            if start_date and end_date and column in filtered_df.columns:
                filtered_df = filtered_df[(filtered_df[column] >= start_date) & (filtered_df[column] <= end_date)]
                filter_description.append(f"{column} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        
        elif filter_type == "categorical" and filter_config.get("selected", False):
            column = filter_config["column"]
            selected_values = filter_config.get("selected_values", [])
            
            if selected_values and column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
                filter_description.append(f"{column} is {', '.join(map(str, selected_values))}")
        
        elif filter_type == "numeric_range" and filter_config.get("selected", False):
            column = filter_config["column"]
            min_value = filter_config.get("min_value")
            max_value = filter_config.get("max_value")
            
            if min_value is not None and max_value is not None and column in filtered_df.columns:
                filtered_df = filtered_df[(filtered_df[column] >= min_value) & (filtered_df[column] <= max_value)]
                filter_description.append(f"{column} between {min_value} and {max_value}")
    
    return filtered_df, filter_description

def generate_dashboard(df, data_overview, selected_visualizations, filters, ai_insights=None):
    """Generate a dashboard with the selected visualizations and filters"""
    st.title("📊 AI-Enhanced Dashboard")
    
    # Display data information
    with st.expander("About this dataset", expanded=False):
        st.markdown(f"**File name:** {st.session_state.file_name}")
        st.markdown(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
        
        if ai_insights:
            st.markdown(f"**Data Domain:** {ai_insights.get('domain', 'Unknown')}")
            st.markdown(f"**Data Description:** {ai_insights.get('description', '')}")
        else:
            st.markdown(f"**Data Description:** {data_overview.get('data_description', '')}")
        
        st.subheader("Sample Data")
        st.dataframe(df.head(5))
    
    # Display AI insights if available
    if ai_insights:
        with st.expander("AI Insights & Recommendations", expanded=True):
            st.markdown("### Key Insights")
            for insight in ai_insights.get("key_insights", []):
                st.markdown(f"- {insight}")
            
            st.markdown("### Key Variables")
            st.markdown(", ".join(ai_insights.get("key_variables", [])))
            
            st.markdown("### Data Quality Issues")
            for issue in ai_insights.get("data_quality_issues", []):
                st.markdown(f"- {issue}")
            
            st.markdown("### Suggested Analyses")
            for suggestion in ai_insights.get("suggested_analyses", []):
                st.markdown(f"- {suggestion}")
    
    # Apply filters
    filter_container = st.container()
    with filter_container:
        st.subheader("Filters")
        
        if not filters:
            st.info("No filters available for this dataset.")
        else:
            filter_cols = st.columns(min(4, max(1, len(filters))))
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
                            # Convert column to datetime if it's not already
                            if df[column].dtype != 'datetime64[ns]':
                                df[column] = pd.to_datetime(df[column], errors='coerce')
                            
                            min_date = df[column].min()
                            max_date = df[column].max()
                            
                            # Ensure min_date and max_date are valid
                            if pd.isna(min_date) or pd.isna(max_date):
                                st.warning(f"No valid dates in {column}")
                                updated_filter["selected"] = False
                                continue
                            
                            # Convert to date for the date picker
                            min_date = min_date.date()
                            max_date = max_date.date()
                            
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
                    
                    elif filter_type == "categorical":
                        try:
                            all_values = filter_config.get("values", [])
                            
                            # Handle potential None values
                            all_values = [v if v is not None else "None" for v in all_values]
                            
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
                            
                            # Convert "None" back to None
                            selected_values = [None if v == "None" else v for v in selected_values]
                            
                            updated_filter["selected_values"] = selected_values
                            updated_filter["selected"] = len(selected_values) > 0
                        except Exception as e:
                            st.error(f"Error with categorical filter for {column}: {str(e)}")
                            updated_filter["selected"] = False
                    
                    elif filter_type == "numeric_range":
                        try:
                            min_val = filter_config.get("min", float(df[column].min()))
                            max_val = filter_config.get("max", float(df[column].max()))
                            
                            # Handle edge cases
                            if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                                if pd.isna(min_val) or pd.isna(max_val):
                                    st.warning(f"No valid numeric data in {column}")
                                else:
                                    st.warning(f"All values in {column} are the same: {min_val}")
                                updated_filter["selected"] = False
                                continue
                            
                            min_value, max_value = st.slider(
                                f"Range",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=(float(min_val), float(max_val)),
                                key=f"range_{column}"
                            )
                            
                            updated_filter["min_value"] = min_value
                            updated_filter["max_value"] = max_value
                            updated_filter["selected"] = min_value > min_val or max_value < max_val
                        except Exception as e:
                            st.error(f"Error with numeric filter for {column}: {str(e)}")
                            updated_filter["selected"] = False
                
                updated_filters.append(updated_filter)
        
            # Apply filters to dataframe
            filtered_df, filter_description = apply_filters(df, updated_filters)
            
            # Show filter summary
            if filter_description:
                st.info(f"Filtered data: {', '.join(filter_description)} (Showing {filtered_df.shape[0]} of {df.shape[0]} rows)")
            else:
                filtered_df = df
                st.info("No filters applied. Showing all data.")
    
    # Dashboard metrics (summary stats)
    metric_container = st.container()
    with metric_container:
        st.subheader("Key Metrics")
        
        # Find numeric columns to use as metrics
        numeric_cols = data_overview.get("numeric_columns", [])
        datetime_cols = data_overview.get("datetime_columns", [])
        
        if numeric_cols:
            metric_cols = st.columns(min(4, len(numeric_cols)))
            
            for i, col in enumerate(numeric_cols[:4]):  # Show up to 4 metrics
                if col in filtered_df.columns:
                    with metric_cols[i]:
                        # Check if we have valid data
                        if filtered_df[col].count() > 0 and not pd.isna(filtered_df[col].mean()):
                            current_value = filtered_df[col].mean()
                            
                            # Calculate delta if we have a time dimension
                            delta = None
                            if datetime_cols and len(datetime_cols) > 0 and filtered_df.shape[0] > 0:
                                time_col = datetime_cols[0]
                                try:
                                    # Make sure time_col is datetime
                                    if filtered_df[time_col].dtype != 'datetime64[ns]':
                                        filtered_df[time_col] = pd.to_datetime(filtered_df[time_col], errors='coerce')
                                    
                                    filtered_df['temp_period'] = filtered_df[time_col].dt.to_period('M')
                                    
                                    # Get the mean value for the most recent and previous period
                                    period_values = filtered_df.groupby('temp_period')[col].mean()
                                    
                                    if len(period_values) >= 2:
                                        current_period = period_values.iloc[-1]
                                        previous_period = period_values.iloc[-2]
                                        
                                        if previous_period != 0:  # Avoid division by zero
                                            delta = ((current_period - previous_period) / previous_period) * 100
                                except Exception as e:
                                    # Silently fail for delta calculation
                                    pass
                            
                            st.metric(
                                label=col,
                                value=f"{current_value:.2f}",
                                delta=f"{delta:.1f}%" if delta is not None else None
                            )
                        else:
                            st.metric(
                                label=col,
                                value="N/A"
                            )
        else:
            st.info("No numeric metrics available for this dataset.")
    
    # Display visualizations in a grid
    st.subheader("Visualizations")
    
    if not selected_visualizations:
        st.info("No visualizations selected. Please choose visualizations from the sidebar.")
    else:
        num_cols = 2  # Number of columns in the grid
        viz_rows = [selected_visualizations[i:i+num_cols] for i in range(0, len(selected_visualizations), num_cols)]
        
        for row in viz_rows:
            cols = st.columns(num_cols)
            
            for i, viz in enumerate(row):
                with cols[i]:
                    st.markdown(f"**{viz['title']}**")
                    success = render_visualization(filtered_df, viz, cols[i])
                    if not success:
                        st.error("Failed to render visualization.")
    
    return True

def main():
    st.title("🧠 AI-Enhanced Data Analyzer & Dashboard Generator")
    st.write("Upload your data file and let AI analyze it and create a custom dashboard for you.")
    
    with st.sidebar:
        st.image("https://img.icons8.com/pulsar-color/96/data-configuration.png", width=80)
        st.header("Data Analysis Options")
        
        # AI integration options
        st.subheader("AI Analysis Settings")
        analysis_option = st.radio(
            "Choose analysis method:",
            ["Standard Analysis", "AI-Enhanced Analysis (OpenAI)"],
            key="analysis_option"
        )
        
        st.session_state.using_openai = (analysis_option == "AI-Enhanced Analysis (OpenAI)")
        st.session_state.using_local_analysis = (analysis_option == "Standard Analysis")
        
        if st.session_state.using_openai:
            # Model selection
            st.session_state.ai_model = st.selectbox(
                "Select AI model:",
                ["gpt-3.5-turbo", "gpt-4o", "gpt-4", "gpt-4-turbo"],
                index=0,
                key="ai_model_select"
            )
            
            # API key input
            api_key = st.text_input(
                "Enter your OpenAI API key:",
                type="password",
                key="api_key_input"
            )
            
            if api_key:
                st.session_state.api_key = api_key
            
            if not st.session_state.api_key:
                st.warning("You need to provide an OpenAI API key for AI-enhanced analysis.")
        
        uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls', 'json', 'txt', 'db', 'sqlite'])
        
        if uploaded_file is not None:
            analyze_button_disabled = (st.session_state.using_openai and not st.session_state.api_key)
            
            if st.button("Analyze Data", type="primary", disabled=analyze_button_disabled):
                with st.spinner("Reading and analyzing data..."):
                    # Reset session state
                    st.session_state.analysis_complete = False
                    st.session_state.dashboard_generated = False
                    st.session_state.ai_insights = None
                    
                    # Read the data file
                    df = read_data_file(uploaded_file)
                    
                    if df is not None and not df.empty:
                        # Store the data
                        st.session_state.data = df
                        
                        # Analyze the data
                        if st.session_state.using_openai and st.session_state.api_key:
                            with st.spinner("AI analyzing your data... This may take a minute..."):
                                # Use OpenAI for enhanced analysis
                                ai_results = analyze_data_with_openai(
                                    df, 
                                    st.session_state.api_key,
                                    st.session_state.ai_model
                                )
                                
                                if ai_results:
                                    st.session_state.ai_insights = ai_results.get("ai_insights")
                                    st.session_state.recommended_visualizations = ai_results.get("recommended_visualizations", [])
                                    st.session_state.filters = ai_results.get("recommended_filters", [])
                                    st.session_state.data_description = ai_results.get("data_description", "")
                                    
                                    # Still run local analysis for column type detection
                                    st.session_state.data_overview = analyze_data(df)
                                    
                                    # Mark analysis as complete
                                    st.session_state.analysis_complete = True
                                    
                                    # Store a copy of the data (for filters)
                                    st.session_state.processed_data = df.copy()
                                else:
                                    st.error("AI analysis failed. Try standard analysis instead.")
                        else:
                            # Use local algorithm for analysis
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
            if st.session_state.ai_insights:
                st.write(st.session_state.ai_insights.get("description", ""))
            else:
                st.write(st.session_state.data_description)
            
            # Select visualizations
            st.subheader("Select Visualizations")
            
            st.session_state.selected_viz_indices = []
            for i, viz in enumerate(st.session_state.recommended_visualizations):
                selected = st.checkbox(viz["title"], value=True, key=f"viz_{i}")
                if selected:
                    st.session_state.selected_viz_indices.append(i)
            
            selected_visualizations = [st.session_state.recommended_visualizations[i] for i in st.session_state.selected_viz_indices]
            
            # Generate dashboard button
            if st.button("Generate Dashboard", type="primary"):
                st.session_state.dashboard_generated = True
                st.rerun()
    
    # Display dashboard content in the main area
    if st.session_state.dashboard_generated and st.session_state.data is not None:
        generate_dashboard(
            st.session_state.processed_data,
            st.session_state.data_overview,
            [st.session_state.recommended_visualizations[i] for i in st.session_state.selected_viz_indices],
            st.session_state.filters,
            st.session_state.ai_insights
        )
    else:
        # Display instructions/welcome message
        st.markdown("""
        ## 👋 Welcome to AI-Enhanced Data Analyzer & Dashboard Generator!
        
        This application helps you instantly analyze your data and create beautiful dashboards with just a few clicks, now enhanced with AI capabilities.
        
        ### How It Works:
        1. **Choose Analysis Method:** Select between standard analysis or AI-enhanced analysis
        2. **Upload your data file** (CSV, Excel, SQLite, etc.)
        3. **Click "Analyze Data"** to examine your dataset
        4. **Select visualizations** you'd like to include
        5. **Generate your dashboard** with a single click
        
        ### Features:
        - **AI-Enhanced Analysis:** Gain deeper insights about your data with AI assistance
        - **Automatic data analysis** with smart visualization recommendations
        - **Interactive filters** to explore your data
        - **Intelligent data detection** to handle various file formats
        - **Key metrics and trends detection**
        - **Easy-to-understand explanations** of findings
        
        ### AI-Enhanced Features:
        - Smarter detection of data patterns and relationships
        - More accurate visualization recommendations
        - Domain-specific insights based on your data
        - Improved data quality assessment
        - Context-aware data interpretation
        
        Upload your file in the sidebar to get started!
        """)

if __name__ == "__main__":
    main()

