
# AI Sales Forecasting & Business Intelligence Dashboard

## Project Overview

This project is an interactive business intelligence dashboard built
with Python and Streamlit.

It analyzes historical sales data, calculates business KPIs, visualizes
regional and category performance, and generates machine-learning-based
sales forecasts.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Random Forest Regression
- Excel and CSV

## Features

1. Upload Excel and CSV sales data.
2. Clean and validate sales records.
3. Calculate total revenue, profit, orders, average order value,
   and profit margin.
4. Analyze monthly sales trends.
5. Compare sales by region, category, and product.
6. Forecast future sales using Random Forest Regression.
7. Visualize historical and forecast sales.
8. Generate automated business insights.
9. Export cleaned data and business reports.
10. Download forecast results in CSV and Excel formats.

## Installation

Install Python 3.10 or newer.

Create a virtual environment:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Sample Dataset

The sample_sales.csv dataset contains sales records across
24 months with order date, region, category, product,
sales, profit, and quantity.

## Machine Learning

Random Forest Regression is used to predict future monthly sales.

The model uses:
- Month number
- Time index
- Previous month sales
- Lagged sales
- Rolling average sales

The forecast is an estimate and should not be treated
as guaranteed future revenue.

## Business Applications

- MIS reporting
- Sales reporting
- Revenue analysis
- Business performance monitoring
- Sales forecasting
- Management decision support

## Future Enhancements

- Forecast model evaluation with a time-based test set
- ARIMA and Prophet model comparison
- Power BI integration
- SQL database connection
- Automated scheduled reports
- LLM-based business insight generation
