
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Sales Forecasting & BI Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Sales Forecasting & Business Intelligence")
st.caption(
    "Analyze historical sales, monitor business KPIs, "
    "and forecast future revenue using machine learning."
)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def normalize_column_names(df):
    df = df.copy()
    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]
    return df


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def clean_sales_data(df):
    df = normalize_column_names(df)

    date_col = find_column(
        df,
        ["order_date", "date", "sales_date", "transaction_date"]
    )

    sales_col = find_column(
        df,
        ["sales", "revenue", "amount", "total_sales", "order_amount"]
    )

    profit_col = find_column(
        df,
        ["profit", "net_profit", "earnings"]
    )

    quantity_col = find_column(
        df,
        ["quantity", "qty", "units", "units_sold"]
    )

    category_col = find_column(
        df,
        ["category", "product_category", "item_category"]
    )

    product_col = find_column(
        df,
        ["product", "product_name", "item", "product_id"]
    )

    region_col = find_column(
        df,
        ["region", "location", "area", "state"]
    )

    order_col = find_column(
        df,
        ["order_id", "transaction_id", "invoice_id"]
    )

    if date_col is None:
        raise ValueError(
            "Date column not found. Expected Order_Date or Date."
        )

    if sales_col is None:
        raise ValueError(
            "Sales column not found. Expected Sales or Revenue."
        )

    original_rows = len(df)

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df[sales_col] = pd.to_numeric(
        df[sales_col],
        errors="coerce"
    )

    if profit_col:
        df[profit_col] = pd.to_numeric(
            df[profit_col],
            errors="coerce"
        )

    if quantity_col:
        df[quantity_col] = pd.to_numeric(
            df[quantity_col],
            errors="coerce"
        )

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Remove records with invalid dates or sales
    df = df.dropna(
        subset=[date_col, sales_col]
    )

    # Keep records with valid numeric sales
    df = df[
        np.isfinite(df[sales_col])
    ]

    df["month"] = df[date_col].dt.to_period(
        "M"
    ).dt.to_timestamp()

    return {
        "data": df,
        "date_col": date_col,
        "sales_col": sales_col,
        "profit_col": profit_col,
        "quantity_col": quantity_col,
        "category_col": category_col,
        "product_col": product_col,
        "region_col": region_col,
        "order_col": order_col,
        "original_rows": original_rows
    }


def create_monthly_sales(df, date_col, sales_col):
    monthly = (
        df.groupby(
            df[date_col].dt.to_period("M")
        )[sales_col]
        .sum()
        .reset_index()
    )

    monthly.columns = ["month", "sales"]

    monthly["month"] = monthly["month"].dt.to_timestamp()

    monthly = monthly.sort_values(
        "month"
    ).reset_index(drop=True)

    # Fill missing months with zero sales
    if not monthly.empty:

        full_months = pd.date_range(
            monthly["month"].min(),
            monthly["month"].max(),
            freq="MS"
        )

        monthly = (
            monthly.set_index("month")
            .reindex(full_months, fill_value=0)
            .rename_axis("month")
            .reset_index()
        )

    return monthly


# =====================================================
# MACHINE LEARNING FORECAST
# =====================================================

def create_forecast(monthly, forecast_months):

    monthly = monthly.copy()

    if len(monthly) < 12:
        raise ValueError(
            "At least 12 months of historical data "
            "are recommended for forecasting."
        )

    monthly = monthly.sort_values(
        "month"
    ).reset_index(drop=True)

    monthly["time_index"] = np.arange(
        len(monthly)
    )

    monthly["month_number"] = (
        monthly["month"].dt.month
    )

    monthly["lag_1"] = monthly["sales"].shift(1)

    monthly["lag_2"] = monthly["sales"].shift(2)

    monthly["lag_3"] = monthly["sales"].shift(3)

    monthly["lag_6"] = monthly["sales"].shift(6)

    monthly["lag_12"] = monthly["sales"].shift(12)

    monthly["rolling_mean_3"] = (
        monthly["sales"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    monthly["rolling_mean_6"] = (
        monthly["sales"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    features = [
        "time_index",
        "month_number",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6"
    ]

    training_data = monthly.dropna(
        subset=features + ["sales"]
    )

    if len(training_data) < 5:
        raise ValueError(
            "Not enough training records after creating lag features."
        )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(
        training_data[features],
        training_data["sales"]
    )

    # Recursive forecasting
    history = monthly["sales"].tolist()

    last_month = monthly["month"].max()

    future_results = []

    for step in range(1, forecast_months + 1):

        next_month = (
            last_month
            + pd.DateOffset(months=step)
        )

        time_index = len(history)

        lag_1 = history[-1]

        lag_2 = history[-2]

        lag_3 = history[-3]

        lag_6 = history[-6]

        lag_12 = history[-12]

        rolling_mean_3 = np.mean(
            history[-3:]
        )

        rolling_mean_6 = np.mean(
            history[-6:]
        )

        input_data = pd.DataFrame([{

            "time_index": time_index,

            "month_number": next_month.month,

            "lag_1": lag_1,

            "lag_2": lag_2,

            "lag_3": lag_3,

            "lag_6": lag_6,

            "lag_12": lag_12,

            "rolling_mean_3": rolling_mean_3,

            "rolling_mean_6": rolling_mean_6

        }])

        predicted_sales = model.predict(
            input_data[features]
        )[0]

        predicted_sales = max(
            0,
            float(predicted_sales)
        )

        history.append(
            predicted_sales
        )

        future_results.append({

            "month": next_month,

            "forecast_sales": predicted_sales

        })

    forecast_df = pd.DataFrame(
        future_results
    )

    return model, forecast_df, training_data, features


# =====================================================
# REPORT EXPORT
# =====================================================

def create_excel_report(
    cleaned_data,
    monthly_sales,
    forecast_data,
    summary_data
):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        cleaned_data.to_excel(
            writer,
            sheet_name="Cleaned_Data",
            index=False
        )

        monthly_sales.to_excel(
            writer,
            sheet_name="Monthly_Sales",
            index=False
        )

        forecast_data.to_excel(
            writer,
            sheet_name="Forecast",
            index=False
        )

        summary_data.to_excel(
            writer,
            sheet_name="Business_Summary",
            index=False
        )

    return output.getvalue()


# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload your sales dataset",
    type=["csv", "xlsx"]
)


if uploaded_file:

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            raw_df = pd.read_csv(
                uploaded_file
            )

        else:

            raw_df = pd.read_excel(
                uploaded_file
            )

        st.success(
            f"Loaded {uploaded_file.name} successfully."
        )

        st.subheader("Raw Data Preview")

        st.dataframe(
            raw_df.head(20),
            use_container_width=True
        )

        processed = clean_sales_data(
            raw_df
        )

        df = processed["data"]

        date_col = processed["date_col"]

        sales_col = processed["sales_col"]

        profit_col = processed["profit_col"]

        quantity_col = processed["quantity_col"]

        category_col = processed["category_col"]

        product_col = processed["product_col"]

        region_col = processed["region_col"]

        order_col = processed["order_col"]

        st.subheader("Data Quality")

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Original Records",
            f"{processed['original_rows']:,}"
        )

        q2.metric(
            "Cleaned Records",
            f"{len(df):,}"
        )

        q3.metric(
            "Removed Records",
            f"{processed['original_rows'] - len(df):,}"
        )

        q4.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}"
        )

        # =================================================
        # BUSINESS KPIs
        # =================================================

        st.divider()

        st.subheader("Business Performance KPIs")

        total_sales = df[sales_col].sum()

        total_profit = (
            df[profit_col].sum()
            if profit_col
            else 0
        )

        total_orders = (
            df[order_col].nunique()
            if order_col
            else len(df)
        )

        average_order_value = (
            total_sales / total_orders
            if total_orders > 0
            else 0
        )

        profit_margin = (
            total_profit / total_sales * 100
            if profit_col and total_sales != 0
            else 0
        )

        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric(
            "Total Sales",
            f"{total_sales:,.2f}"
        )

        k2.metric(
            "Total Profit",
            f"{total_profit:,.2f}"
            if profit_col
            else "N/A"
        )

        k3.metric(
            "Orders",
            f"{total_orders:,}"
        )

        k4.metric(
            "Average Order Value",
            f"{average_order_value:,.2f}"
        )

        k5.metric(
            "Profit Margin",
            f"{profit_margin:.2f}%"
            if profit_col
            else "N/A"
        )

        # =================================================
        # MONTHLY SALES TREND
        # =================================================

        st.divider()

        st.subheader("Monthly Sales Trend")

        monthly_sales = create_monthly_sales(
            df,
            date_col,
            sales_col
        )

        monthly_chart = px.line(

            monthly_sales,

            x="month",

            y="sales",

            markers=True,

            title="Monthly Revenue Trend"

        )

        st.plotly_chart(
            monthly_chart,
            use_container_width=True
        )

        # =================================================
        # CATEGORY ANALYSIS
        # =================================================

        if category_col:

            st.subheader("Sales by Category")

            category_sales = (
                df.groupby(category_col)[sales_col]
                .sum()
                .reset_index()
                .sort_values(
                    sales_col,
                    ascending=False
                )
            )

            category_chart = px.bar(

                category_sales,

                x=category_col,

                y=sales_col,

                title="Revenue by Category",

                text_auto=".2s"

            )

            st.plotly_chart(
                category_chart,
                use_container_width=True
            )

        # =================================================
        # REGION ANALYSIS
        # =================================================

        if region_col:

            st.subheader("Sales by Region")

            region_sales = (
                df.groupby(region_col)[sales_col]
                .sum()
                .reset_index()
                .sort_values(
                    sales_col,
                    ascending=False
                )
            )

            region_chart = px.bar(

                region_sales,

                x=region_col,

                y=sales_col,

                title="Regional Revenue Comparison",

                text_auto=".2s"

            )

            st.plotly_chart(
                region_chart,
                use_container_width=True
            )

        # =================================================
        # PRODUCT ANALYSIS
        # =================================================

        if product_col:

            st.subheader("Top Performing Products")

            product_sales = (
                df.groupby(product_col)[sales_col]
                .sum()
                .reset_index()
                .sort_values(
                    sales_col,
                    ascending=False
                )
                .head(10)
            )

            product_chart = px.bar(

                product_sales,

                x=product_col,

                y=sales_col,

                title="Top 10 Products by Revenue",

                text_auto=".2s"

            )

            st.plotly_chart(
                product_chart,
                use_container_width=True
            )

        # =================================================
        # SALES FORECASTING
        # =================================================

        st.divider()

        st.subheader(
            "Machine Learning Sales Forecast"
        )

        st.write("""
        The Random Forest regression model uses historical monthly sales,
        previous-month sales values, rolling averages, and month number
        to estimate future sales.

        Forecasts are estimates, not guaranteed future results.
        """)

        forecast_months = st.slider(
            "Number of months to forecast",
            min_value=1,
            max_value=12,
            value=3
        )

        if st.button(
            "Generate Sales Forecast",
            type="primary"
        ):

            try:

                model, forecast_df, training_data, features = (
                    create_forecast(
                        monthly_sales,
                        forecast_months
                    )
                )

                st.session_state["forecast_df"] = forecast_df

                st.session_state["forecast_model"] = model

                st.success(
                    "Forecast generated successfully."
                )

            except Exception as error:

                st.error(
                    str(error)
                )

        if "forecast_df" in st.session_state:

            forecast_df = st.session_state[
                "forecast_df"
            ]

            st.subheader("Forecast Results")

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            combined_chart = go.Figure()

            combined_chart.add_trace(
                go.Scatter(
                    x=monthly_sales["month"],
                    y=monthly_sales["sales"],
                    mode="lines+markers",
                    name="Historical Sales"
                )
            )

            combined_chart.add_trace(
                go.Scatter(
                    x=forecast_df["month"],
                    y=forecast_df["forecast_sales"],
                    mode="lines+markers",
                    name="Forecast Sales",
                    line=dict(
                        dash="dash"
                    )
                )
            )

            combined_chart.update_layout(
                title="Historical Sales vs Forecast",
                xaxis_title="Month",
                yaxis_title="Sales",
                hovermode="x unified"
            )

            st.plotly_chart(
                combined_chart,
                use_container_width=True
            )

            expected_future_sales = (
                forecast_df["forecast_sales"].sum()
            )

            st.metric(
                f"Expected Sales for Next {forecast_months} Months",
                f"{expected_future_sales:,.2f}"
            )

        # =================================================
        # AUTOMATED BUSINESS INSIGHTS
        # =================================================

        st.divider()

        st.subheader(
            "Automated Business Insights"
        )

        insights = []

        insights.append(
            f"Total sales recorded: {total_sales:,.2f}."
        )

        insights.append(
            f"Total records analyzed: {len(df):,}."
        )

        if len(monthly_sales) >= 2:

            first_sales = monthly_sales[
                "sales"
            ].iloc[0]

            last_sales = monthly_sales[
                "sales"
            ].iloc[-1]

            if first_sales != 0:

                change = (
                    (last_sales - first_sales)
                    / abs(first_sales)
                ) * 100

                insights.append(
                    f"Sales changed by {change:.2f}% "
                    "between the first and last recorded months."
                )

        if category_col:

            top_category = category_sales.iloc[0][
                category_col
            ]

            insights.append(
                f"Highest revenue category: {top_category}."
            )

        if region_col:

            top_region = region_sales.iloc[0][
                region_col
            ]

            insights.append(
                f"Highest revenue region: {top_region}."
            )

        if profit_col:

            insights.append(
                f"Total profit: {total_profit:,.2f}."
            )

            insights.append(
                f"Profit margin: {profit_margin:.2f}%."
            )

        for insight in insights:

            st.write(
                "• " + insight
            )

        # =================================================
        # BUSINESS SUMMARY TABLE
        # =================================================

        summary_data = pd.DataFrame({

            "Metric": [

                "Total Sales",

                "Total Profit",

                "Total Orders",

                "Average Order Value",

                "Profit Margin",

                "Total Records"

            ],

            "Value": [

                total_sales,

                total_profit if profit_col else np.nan,

                total_orders,

                average_order_value,

                profit_margin if profit_col else np.nan,

                len(df)

            ]

        })

        # =================================================
        # DOWNLOAD REPORT
        # =================================================

        st.divider()

        st.subheader("Download Reports")

        cleaned_csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Cleaned Sales CSV",
            data=cleaned_csv,
            file_name="cleaned_sales.csv",
            mime="text/csv"
        )

        monthly_csv = monthly_sales.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Monthly Sales Report",
            data=monthly_csv,
            file_name="monthly_sales_report.csv",
            mime="text/csv"
        )

        if "forecast_df" in st.session_state:

            forecast_csv = st.session_state[
                "forecast_df"
            ].to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Sales Forecast CSV",
                data=forecast_csv,
                file_name="sales_forecast.csv",
                mime="text/csv"
            )

        excel_report = create_excel_report(
            df,
            monthly_sales,
            st.session_state.get(
                "forecast_df",
                pd.DataFrame()
            ),
            summary_data
        )

        st.download_button(
            "Download Complete Excel Business Report",
            data=excel_report,
            file_name="sales_business_report.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

        st.success(
            "Sales analysis completed successfully."
        )

    except Exception as error:

        st.error(
            f"Unable to process the dataset: {error}"
        )

        st.info("""
        Check that your dataset contains a date column and
        a numeric sales or revenue column.

        Optional columns include Profit, Category, Product,
        Region, Quantity, and Order_ID.
        """)

else:

    st.info(
        "Upload a sales CSV or Excel dataset to begin analysis."
    )
