import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import streamlit as st

def predict_month_end(df):
    """
    Predicts the total spending by the end of the month using Linear Regression.
    Robustly handles currency symbols (₹, $, etc.) and commas.
    """
    
    # 1. Filter for Expenses Only
    # Ensure we don't crash if 'Category' is missing
    if 'Category' not in df.columns:
        st.error("Category column missing. Please categorize data first.")
        return None, 0

    expense_df = df[~df['Category'].isin(['Income', 'Transfer', 'Other'])].copy()
    
    # --- FIX START: ROBUST CLEANING ---
    # Regex: Remove anything that is NOT a digit (\d) or a dot (.)
    # This handles '₹761.25', '$ 1,000.00', 'EUR 50', etc.
    expense_df['Amount'] = expense_df['Amount'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    
    # Convert to float
    expense_df['Amount'] = pd.to_numeric(expense_df['Amount'], errors='coerce')
    
    # Drop rows where Amount became NaN (garbage data)
    expense_df = expense_df.dropna(subset=['Amount'])
    
    if expense_df.empty:
        st.warning("No valid expense data found after cleaning. Check if amounts are formatted correctly.")
        return None, 0
    # --- FIX END ---

    # 2. Group by Date
    # Try multiple date formats to be safe
    expense_df['Date'] = pd.to_datetime(expense_df['Date'], dayfirst=True, errors='coerce')
    
    # Drop invalid dates
    expense_df = expense_df.dropna(subset=['Date'])
    
    daily_spend = expense_df.groupby('Date')['Amount'].sum().reset_index()
    
    if len(daily_spend) < 2:
        st.warning("Not enough daily data points to forecast (need at least 2 days).")
        return None, 0

    # 3. Calculate Cumulative Spending
    daily_spend = daily_spend.sort_values('Date')
    daily_spend['Cumulative'] = daily_spend['Amount'].cumsum()
    
    # 4. Prepare Regression
    daily_spend['Day'] = daily_spend['Date'].dt.day
    X = daily_spend[['Day']]
    y = daily_spend['Cumulative']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 5. Predict
    days_in_month = daily_spend['Date'].dt.days_in_month.max()
    future_days = np.array(range(1, days_in_month + 1)).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    # 6. Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_spend['Day'], y=daily_spend['Cumulative'], mode='lines+markers', name='Actual Spending'))
    fig.add_trace(go.Scatter(x=future_days.flatten(), y=predictions, mode='lines', name='Predicted Trend', line=dict(dash='dash')))
    
    predicted_total = predictions[-1]
    
    fig.update_layout(title=f"Projected Month-End: ₹{predicted_total:,.2f}", xaxis_title="Day", yaxis_title="Cumulative Amount")
    
    return fig, predicted_total