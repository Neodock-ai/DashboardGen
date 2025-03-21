# AI Data Analyzer & Dashboard Generator

This Streamlit application automatically analyzes data files, understands what the data is about, and creates a full-fledged interactive dashboard with appropriate visualizations and filters.

## Features

- **Upload any data file** (CSV, Excel, SQLite DB, etc.)
- **AI-powered data analysis** that understands your data context and meaning
- **Automatic visualization recommendations** based on data characteristics
- **Interactive filtering** to explore your data
- **Key metrics** automatically identified and displayed
- **Responsive dashboard** with explanations of insights

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and go to `http://localhost:8501`

## Deploying to Streamlit Cloud

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```

2. Go to [Streamlit Sharing](https://share.streamlit.io/) and sign in with your GitHub account.

3. Click on "New app" and select your repository, branch, and main file path (`app.py`).

4. Click "Deploy" and wait for the deployment to complete.

5. Your app will be available at `https://share.streamlit.io/YOUR_USERNAME/YOUR_REPO/main/app.py`

## How It Works

1. **Data Ingestion**: The app reads various file formats and converts them to pandas DataFrames.
2. **Data Analysis**: Multiple analyses are performed to understand data types, relationships, and patterns.
3. **Visualization Recommendation**: Based on the analysis, appropriate visualizations are suggested.
4. **Filter Generation**: Interactive filters are created based on column types.
5. **Dashboard Assembly**: A complete dashboard is assembled with all components.

## File Structure

- `app.py`: The main Streamlit application
- `requirements.txt`: List of required Python packages
- `README.md`: This documentation file

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## License

MIT
