import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os
#import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Excel file path
path = "expenses_earnings.xlsx"

# Page configuration
st.set_page_config(
    page_title="Advanced Financial Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced data loading with type conversion
@st.cache_data
def load_data(path):
    try:
        if os.path.exists(path):
            df = pd.read_excel(path)
            required_columns = [
                "date", "descriptions", "main_category", "sub_category",
                "expense_type", "payment_mode", "openning_balance",
                "earnings_amount", "invesment_amount", "expenses_amount"
            ]
            
            # Data validation
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                st.warning(f"Missing columns in data: {', '.join(missing)}. Using default structure.")
                return pd.DataFrame(columns=required_columns)
            
            # Convert numeric columns
            numeric_cols = ['openning_balance', 'earnings_amount', 'invesment_amount', 'expenses_amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Date processing
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
                df['month'] = pd.to_datetime(df['date']).dt.strftime("%B %Y")
                df['year'] = pd.to_datetime(df['date']).dt.year
                df['month_num'] = pd.to_datetime(df['date']).dt.month
                df['quarter'] = pd.to_datetime(df['date']).dt.quarter
                df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
                df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
            
            # Calculate closing balance
            if all(col in df.columns for col in ['openning_balance', 'earnings_amount', 'invesment_amount', 'expenses_amount']):
                df['closing_balance'] = df['openning_balance'].astype(float) + \
                                       df['earnings_amount'].astype(float) + \
                                       df['invesment_amount'].astype(float) - \
                                       df['expenses_amount'].astype(float)
            
            return df
        else:
            st.info("No existing data file found. Creating new dataframe.")
            return pd.DataFrame(columns=required_columns)
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return pd.DataFrame(columns=required_columns)

# Save function with backup
def save_data(df, path):
    try:
        # Create backup if file exists
        if os.path.exists(path):
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(backup_path, index=False)
            st.info(f"Created backup: {backup_path}")
        
        df.to_excel(path, index=False)
        st.success("Data saved successfully!")
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

# Initialize data
if 'df' not in st.session_state:
    st.session_state.df = load_data(path)

# Financial health calculation
def calculate_financial_health(df):
    metrics = {}
    
    if not df.empty:
        # Convert to numeric to ensure calculations work
        numeric_cols = ['openning_balance','earnings_amount', 'expenses_amount', 'invesment_amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Basic metrics
        metrics['total_earnings'] = df['earnings_amount'].sum()
        metrics['total_expenses'] = df['expenses_amount'].sum()
        metrics['total_investment'] = df['invesment_amount'].sum()
        metrics['net_profit'] = metrics['total_earnings'] + metrics['total_investment'] - metrics['total_expenses']
        
        # Ratios
        metrics['savings_rate'] = (metrics['net_profit'] / metrics['total_earnings']) * 100 if metrics['total_earnings'] > 0 else 0
        metrics['expense_ratio'] = (metrics['total_expenses'] / metrics['total_earnings']) * 100 if metrics['total_earnings'] > 0 else 0
        metrics['investment_ratio'] = (metrics['total_investment'] / metrics['total_earnings']) * 100 if metrics['total_earnings'] > 0 else 0
        
        # Monthly averages
        if 'month' in df.columns:
            monthly_data = df.groupby('month').agg({
                'earnings_amount': 'sum',
                'expenses_amount': 'sum',
                'invesment_amount': 'sum'
            }).reset_index()
            
            if not monthly_data.empty:
                metrics['avg_monthly_earnings'] = monthly_data['earnings_amount'].mean()
                metrics['avg_monthly_expenses'] = monthly_data['expenses_amount'].mean()
                metrics['avg_monthly_savings'] = metrics['avg_monthly_earnings'] - metrics['avg_monthly_expenses']
    
    return metrics


def forecast_finances(df, periods=6):
    """Generate financial forecasts using linear regression"""
    if df.empty or 'date' not in df.columns:
        st.warning("Not enough data for forecasting")
        return None
    
    try:
        # Prepare time series data - ensure numeric values
        numeric_cols = ['earnings_amount', 'expenses_amount', 'invesment_amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Group by date and resample to monthly
        ts_data = df.groupby(pd.to_datetime(df['date'])).agg({
            'earnings_amount': 'sum',
            'expenses_amount': 'sum',
            'invesment_amount': 'sum'
        }).resample('M').sum().reset_index()
        
        if len(ts_data) < 3:  # Need at least 3 points for forecasting
            st.warning("Need at least 3 months of data for forecasting")
            return None
        
        # Create future dates
        last_date = ts_data['date'].max()
        future_dates = [last_date + relativedelta(months=i) for i in range(1, periods+1)]
        
        # Prepare forecasting data
        forecasts = {}
        X = np.array(range(len(ts_data))).reshape(-1, 1)
        
        for col in numeric_cols:
            if col in ts_data.columns:
                y = ts_data[col].values
                
                # Handle cases where all values are zero
                if np.all(y == 0):
                    forecasts[col] = np.zeros(periods)
                    continue
                    
                model = LinearRegression().fit(X, y)
                future_X = np.array(range(len(ts_data), len(ts_data)+periods)).reshape(-1, 1)
                forecasts[col] = model.predict(future_X)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'type': 'forecast'
        })
        
        for col in forecasts:
            forecast_df[col] = forecasts[col]
        
        # Combine with historical data
        ts_data['type'] = 'actual'
        combined = pd.concat([ts_data, forecast_df])
        
        return combined
        
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")
        return None

def forecasting_page(filtered_df):
    st.title("🔮 Financial Forecasting")
    
    if filtered_df.empty:
        st.warning("No data available for forecasting")
        return
    
    # Forecasting parameters
    periods = st.slider("Forecast Period (months)", 1, 12, 6, key="forecast_period")
    
    # Generate forecast
    forecast_data = forecast_finances(filtered_df, periods)
    
    if forecast_data is None:
        return
    
    # Plot forecast
    fig = px.line(forecast_data, 
                 x='date', 
                 y=['earnings_amount', 'expenses_amount', 'invesment_amount'],
                 color='type', 
                 title=f"{periods}-Month Financial Forecast",
                 labels={'value': 'Amount (৳)', 'date': 'Date'},
                 line_dash='type')
    
    fig.update_layout(
        hovermode='x unified',
        legend_title_text='Type'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast summary
    st.subheader("Forecast Summary")
    
    # Get last actual and forecast periods
    last_actual = forecast_data[forecast_data['type'] == 'actual'].iloc[-1]
    forecast_period = forecast_data[forecast_data['type'] == 'forecast']
    
    if not forecast_period.empty:
        col1, col2, col3 = st.columns(3)
        
        # Earnings projection
        earnings_change = 0
        if last_actual['earnings_amount'] != 0:
            earnings_change = (forecast_period['earnings_amount'].sum() / last_actual['earnings_amount'] - 1)*100
            
        col1.metric("Projected Earnings", 
                   f"৳{forecast_period['earnings_amount'].sum():,.2f}",
                   delta=f"{earnings_change:.1f}% vs last period")
        
        # Expenses projection
        expenses_change = 0
        if last_actual['expenses_amount'] != 0:
            expenses_change = (forecast_period['expenses_amount'].sum() / last_actual['expenses_amount'] - 1)*100
            
        col2.metric("Projected Expenses", 
                   f"৳{forecast_period['expenses_amount'].sum():,.2f}",
                   delta=f"{expenses_change:.1f}% vs last period")
        
        # Investment projection
        investment_change = 0
        if last_actual['invesment_amount'] != 0:
            investment_change = (forecast_period['invesment_amount'].sum() / last_actual['invesment_amount'] - 1)*100
            
        col3.metric("Projected Investment", 
                   f"৳{forecast_period['invesment_amount'].sum():,.2f}",
                   delta=f"{investment_change:.1f}% vs last period")
        
        # Detailed forecast table
        st.subheader("Detailed Forecast")
        st.dataframe(forecast_data.set_index('date').sort_index(ascending=False), 
                    use_container_width=True)

# Sidebar navigation with proper key management
def sidebar_navigation():
    st.sidebar.title("📊 Financial Analytics")
    st.sidebar.header("🧭 Navigation")
    
    menu_options = {
        "🏠 Home": home_page,
        "🏠 Dashboard": dashboard_page,
        "📝 Data Entry": data_entry_page,
        "📊 Financial Reports": financial_reports_page,
        "📈 Trend Analysis": trend_analysis_page,
        "🔮 Forecasting": forecasting_page,
        "💰 Budget Planning": budget_planning_page,
        "📉 Performance Metrics": performance_metrics_page,
        "✏️ Data Management": data_management_page,
        "👋 Hello Sir": managing_director_page,
    }
    
    selected = st.sidebar.radio(
        "Go to", 
        list(menu_options.keys()),
        key="main_navigation_radio"
    )
    
    # Apply filters
    filtered_df = st.session_state.df.copy()
    if not filtered_df.empty and 'date' in filtered_df.columns:
        min_date = filtered_df['date'].min()
        max_date = filtered_df['date'].max()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key="date_range_filter"
        )
        
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date'] >= date_range[0]) & 
                (filtered_df['date'] <= date_range[1])
            ]
    
    if not filtered_df.empty and 'expense_type' in filtered_df.columns:
        categories = filtered_df['expense_type'].unique()
        selected_categories = st.sidebar.multiselect(
            "Filter by Expense Type",
            categories,
            default=categories,
            key="category_filter"
        )
        
        if selected_categories:
            filtered_df = filtered_df[filtered_df['expense_type'].isin(selected_categories)]
    
    # Call the selected page function with filtered_df
    menu_options[selected](filtered_df)


# Home page
def home_page(filtered_df):
    st.title("📊 Financial Dashboard")
    st.write("Comprehensive overview of your financial performance")
    
    metrics = calculate_financial_health(filtered_df)
    
    # [Previous metric cards and health indicators sections remain the same...]
    
    # Three-Level Category Analysis Section
    st.subheader("🗂 Three-Level Category Analysis")
    
    if not filtered_df.empty and all(col in filtered_df.columns for col in ['main_category', 'sub_category', 'expense_type']):
        tab1, tab2, tab3 = st.tabs(["Main Categories", "Sub-Categories", "Expense Types"])
        
        with tab1:
            st.markdown("### Main Categories Summary")
            main_cat_data = filtered_df.groupby('main_category').agg({
                'earnings_amount': 'sum',
                'expenses_amount': 'sum',
                'invesment_amount': 'sum'
            }).reset_index()
            
            # Visualization for main categories
            fig = px.sunburst(
                filtered_df,
                path=['main_category'],
                values='expenses_amount',
                title="Expenses by Main Category"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed data table
            with st.expander("View Main Category Details"):
                st.dataframe(
                    main_cat_data.sort_values('expenses_amount', ascending=False),
                    use_container_width=True,
                    column_config={
                        "main_category": "Main Category",
                        "expenses_amount": st.column_config.NumberColumn("Expenses", format="৳%.2f"),
                        "earnings_amount": st.column_config.NumberColumn("Earnings", format="৳%.2f"),
                        "invesment_amount": st.column_config.NumberColumn("Investment", format="৳%.2f")
                    }
                )
        
        with tab2:
            st.markdown("### Sub-Categories Breakdown")
            subcat_data = filtered_df.groupby(['main_category', 'sub_category']).agg({
                'expenses_amount': 'sum'
            }).reset_index()
            
            # Tree map visualization
            fig = px.treemap(
                subcat_data,
                path=['main_category', 'sub_category'],
                values='expenses_amount',
                title="Expenses by Sub-Category"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed data table
            with st.expander("View Sub-Category Details"):
                st.dataframe(
                    subcat_data.sort_values('expenses_amount', ascending=False),
                    use_container_width=True,
                    column_config={
                        "main_category": "Main Category",
                        "sub_category": "Sub Category",
                        "expenses_amount": st.column_config.NumberColumn("Expenses", format="৳%.2f")
                    }
                )
        
        with tab3:
            st.markdown("### Expense Types Analysis")
            expense_type_data = filtered_df.groupby(['main_category', 'sub_category', 'expense_type']).agg({
                'expenses_amount': 'sum'
            }).reset_index()
            
            # Interactive bar chart
            fig = px.bar(
                expense_type_data,
                x='expense_type',
                y='expenses_amount',
                color='main_category',
                title="Expenses by Type Across Categories",
                labels={'expenses_amount': 'Amount (৳)', 'expense_type': 'Expense Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed hierarchical view
            st.markdown("#### Hierarchical View")
            fig = px.sunburst(
                expense_type_data,
                path=['main_category', 'sub_category', 'expense_type'],
                values='expenses_amount',
                title="Expense Type Hierarchy"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed data table
            with st.expander("View Expense Type Details"):
                st.dataframe(
                    expense_type_data.sort_values('expenses_amount', ascending=False),
                    use_container_width=True,
                    column_config={
                        "main_category": "Main Category",
                        "sub_category": "Sub Category",
                        "expense_type": "Expense Type",
                        "expenses_amount": st.column_config.NumberColumn("Expenses", format="৳%.2f")
                    }
                )
    else:
        st.warning("Category data not available for analysis")

    # [Rest of your existing dashboard sections remain unchanged...]
# Dashboard page
def dashboard_page(filtered_df):
    st.title("📊 Financial Dashboard")
    st.write("Comprehensive overview of your financial performance")
    
    metrics = calculate_financial_health(filtered_df)
    
    # Helper function to create consistent metric cards
    def create_metric_card(col, title, value, delta=None, help_text=None):
        with col:
            st.metric(
                title,
                value=f"৳{value:,.2f}" if isinstance(value, (int, float)) else value,
                delta=delta,
                help=help_text
            )
    
    # Top Metrics Section
    st.subheader("🔢 Key Financial Metrics")
    col1, col2, col3, col4 = st.columns(4)
    create_metric_card(col1, "Total Earnings", metrics.get('total_earnings', 0), 
                      f"{metrics.get('savings_rate', 0):.1f}% Savings Rate")
    create_metric_card(col2, "Total Expenses", metrics.get('total_expenses', 0),
                      f"{metrics.get('expense_ratio', 0):.1f}% of Earnings")
    create_metric_card(col3, "Net Profit", metrics.get('net_profit', 0),
                      "Surplus" if metrics.get('net_profit', 0) >= 0 else "Deficit")
    create_metric_card(col4, "Total Investment", metrics.get('total_investment', 0),
                      f"{metrics.get('investment_ratio', 0):.1f}% of Earnings")
    
    # Financial Health Indicators
    st.subheader("📉 Financial Health Indicators")
    health_col1, health_col2, health_col3 = st.columns(3)
    
    # Savings Rate Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=metrics.get('savings_rate', 0),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Savings Rate (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "red"},
                {'range': [20, 40], 'color': "orange"},
                {'range': [40, 100], 'color': "green"}
            ]
        }
    ))
    health_col1.plotly_chart(fig, use_container_width=True)
    
    create_metric_card(health_col2, "Expense Ratio", 
                      f"{metrics.get('expense_ratio', 0):.1f}%",
                      help_text="Expenses as percentage of earnings")
    create_metric_card(health_col3, "Investment Ratio",
                      f"{metrics.get('investment_ratio', 0):.1f}%",
                      help_text="Investment as percentage of earnings")
    
    # Helper function for financial visualizations
    def create_financial_visualization(data, group_by, metrics_list, title, chart_type='line'):
        if data.empty or group_by not in data.columns:
            st.warning(f"No {group_by} data available for {title.lower()}")
            return None
        
        grouped_data = data.groupby(group_by).agg({m: 'sum' for m in metrics_list}).reset_index()
        
        if chart_type == 'line':
            fig = px.line(grouped_data, x=group_by, y=metrics_list,
                         title=title, labels={'value': 'Amount (৳)'})
        elif chart_type == 'bar':
            fig = px.bar(grouped_data, x=group_by, y=metrics_list,
                         title=title, labels={'value': 'Amount (৳)'})
        elif chart_type == 'pie':
            fig = px.pie(grouped_data, names=group_by, values=metrics_list[0],
                        title=title)
        
        st.plotly_chart(fig, use_container_width=True)
        with st.expander(f"View {title} Data"):
            st.dataframe(grouped_data, use_container_width=True)
        return grouped_data
    
    # Financial Trends and Analyses
    financial_metrics = ['earnings_amount', 'expenses_amount', 'invesment_amount']
    
    # Monthly Summary
    st.subheader("📅 Monthly Summary")
    monthly_data = create_financial_visualization(
        filtered_df, 'month', financial_metrics, 
        "Monthly Financial Summary", 'bar'
    )
    
    # Category Breakdown
    st.subheader("📊 Category Breakdown")
    category_data = create_financial_visualization(
        filtered_df, 'main_category', ['expenses_amount'],
        "Expenses by Main Category", 'pie'
    )
    
    # Payment Mode Analysis
    st.subheader("💳 Payment Mode Analysis")
    payment_data = create_financial_visualization(
        filtered_df, 'payment_mode', ['earnings_amount', 'expenses_amount'],
        "Transactions by Payment Mode", 'bar'
    )
    
    # NEW: Expense Type Summary
    st.subheader("🧾 Expense Type Summary")
    if not filtered_df.empty and 'expense_type' in filtered_df.columns:
        # Calculate expense type distribution
        expense_type_data = filtered_df.groupby('expense_type').agg({
            'expenses_amount': 'sum'
        }).reset_index().sort_values('expenses_amount', ascending=False)
        
        # Create two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for expense type distribution
            fig = px.pie(expense_type_data, 
                        names='expense_type', 
                        values='expenses_amount',
                        title="Expense Distribution by Type",
                        hole=0.3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart for expense type comparison
            fig = px.bar(expense_type_data,
                        x='expense_type',
                        y='expenses_amount',
                        title="Expenses by Type",
                        labels={'expenses_amount': 'Amount (৳)', 'expense_type': 'Expense Type'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("View Expense Type Details"):
            st.dataframe(expense_type_data, use_container_width=True)
    else:
        st.warning("No expense type data available")
    
    # Financial Trend Over Time
    st.subheader("📈 Financial Trend")
    trend_data = create_financial_visualization(
        filtered_df, 'date', financial_metrics,
        "Financial Trend Over Time", 'line'
    )
    
    # Recent Transactions
    st.subheader("🔄 Recent Transactions")
    if not filtered_df.empty and 'date' in filtered_df.columns:
        st.dataframe(
            filtered_df.sort_values('date', ascending=False).head(10), 
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "earnings_amount": st.column_config.NumberColumn("Earnings", format="৳%.2f"),
                "expenses_amount": st.column_config.NumberColumn("Expenses", format="৳%.2f"),
                "invesment_amount": st.column_config.NumberColumn("Investment", format="৳%.2f"),
                "expense_type": "Expense Type"  # Added expense_type to displayed columns
            }
        )
    else:
        st.warning("No transaction data available")# Data Entry Page

def data_entry_page(filtered_df):
    st.title("📝 Data Entry")
    
    with st.form("entry_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            entry_date = st.date_input("Date", date.today())
            description = st.text_input("Description")
            main_category = st.selectbox("Main Category", ["Revenue", "Expense", "Investment", "Other"])
            sub_category = st.text_input("Sub Category")
            
        with col2:
            payment_mode = st.selectbox("Payment Mode", ["Cash", "Bank Transfer", "Credit Card", "Mobile Banking"])
            earnings = st.number_input("Earnings Amount (৳)", min_value=0.0, value=0.0)
            investment = st.number_input("Investment Amount (৳)", min_value=0.0, value=0.0)
            expenses = st.number_input("Expenses Amount (৳)", min_value=0.0, value=0.0)
        
        submitted = st.form_submit_button("Save Entry")
        
        if submitted:
            new_entry = {
                "date": entry_date,
                "descriptions": description,
                "main_category": main_category,
                "sub_category": sub_category,
                "payment_mode": payment_mode,
                "earnings_amount": earnings,
                "invesment_amount": investment,
                "expenses_amount": expenses,
                "openning_balance": 0  # Will be calculated
            }
            
            # Update dataframe
            new_df = pd.concat([st.session_state.df, pd.DataFrame([new_entry])], ignore_index=True)
            
            # Recalculate opening balances
            new_df = new_df.sort_values('date')
            new_df['openning_balance'] = new_df['earnings_amount'].cumsum() - new_df['expenses_amount'].cumsum()
            
            if save_data(new_df, path):
                st.session_state.df = new_df
                st.success("Entry saved successfully!")
            else:
                st.error("Failed to save entry.")

def financial_reports_page(filtered_df):  # Add parameter here
    st.title("📊 Financial Reports")
    
    report_type = st.selectbox("Select Report Type", [
        "Category Breakdown",
        "Payment Mode Analysis",
        "Monthly Summary",
        "Annual Summary"
    ], key="report_type_select")
    
    if report_type == "Category Breakdown":
        st.subheader("📊 Category-wise Breakdown")
        
        tab1, tab2 = st.tabs(["Main Category", "Sub Category"])
        
        with tab1:
            if not filtered_df.empty and 'main_category' in filtered_df.columns:
                cat_data = filtered_df.groupby('main_category').agg({
                    'earnings_amount': 'sum',
                    'expenses_amount': 'sum',
                    'invesment_amount': 'sum'
                }).reset_index()
                
                fig = px.pie(cat_data, names='main_category', values='expenses_amount', 
                            title="Expenses by Main Category")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(cat_data, use_container_width=True)
        
        with tab2:
            if not filtered_df.empty and 'sub_category' in filtered_df.columns:
                subcat_data = filtered_df.groupby(['main_category', 'sub_category']).agg({
                    'expenses_amount': 'sum'
                }).reset_index().sort_values('expenses_amount', ascending=False)
                
                fig = px.treemap(subcat_data, path=['main_category', 'sub_category'], values='expenses_amount',
                                title="Expenses by Sub Category")
                st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Payment Mode Analysis":
        st.subheader("💳 Payment Mode Analysis")
        
        if not filtered_df.empty and 'payment_mode' in filtered_df.columns:
            pm_data = filtered_df.groupby('payment_mode').agg({
                'earnings_amount': 'sum',
                'expenses_amount': 'sum'
            }).reset_index()
            
            fig = px.bar(pm_data, x='payment_mode', y=['earnings_amount', 'expenses_amount'],
                        barmode='group', title="Transactions by Payment Mode")
            st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Monthly Summary":
        st.subheader("📅 Monthly Summary")
        
        if not filtered_df.empty and 'month' in filtered_df.columns:
            monthly_data = filtered_df.groupby('month').agg({
                'earnings_amount': 'sum',
                'expenses_amount': 'sum',
                'invesment_amount': 'sum'
            }).reset_index()
            
            fig = px.line(monthly_data, x='month', y=['earnings_amount', 'expenses_amount'],
                          title="Monthly Trend", labels={'value': 'Amount (৳)'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly comparison
            pivot_data = monthly_data.copy()
            pivot_data['month_name'] = pd.to_datetime(pivot_data['month']).dt.strftime("%B")
            pivot_data['year'] = pd.to_datetime(pivot_data['month']).dt.year
            
            fig = px.bar(pivot_data, x='month_name', y='expenses_amount', color='year',
                         barmode='group', title="Monthly Comparison by Year")
            st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Annual Summary":
        st.subheader("📅 Annual Summary")
        
        if not filtered_df.empty and 'year' in filtered_df.columns:
            annual_data = filtered_df.groupby('year').agg({
                'earnings_amount': 'sum',
                'expenses_amount': 'sum',
                'invesment_amount': 'sum'
            }).reset_index()
            
            fig = px.bar(annual_data, x='year', y=['earnings_amount', 'expenses_amount'],
                        barmode='group', title="Annual Performance")
            st.plotly_chart(fig, use_container_width=True)


def trend_analysis_page(filtered_df):  # Add parameter here
    st.title("📈 Trend Analysis")
    
    if filtered_df.empty or 'date' not in filtered_df.columns:
        st.warning("No data available for trend analysis")
        return
    
    
    # Time period selection
    period = st.selectbox("Analysis Period", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
    
    # Prepare data based on selected period
    if period == "Daily":
        ts_data = filtered_df.groupby('date').agg({
            'earnings_amount': 'sum',
            'expenses_amount': 'sum'
        }).reset_index()
        x_axis = 'date'
    elif period == "Weekly":
        ts_data = filtered_df.copy()
        ts_data['week'] = pd.to_datetime(ts_data['date']).dt.strftime('%Y-W%U')
        ts_data = ts_data.groupby('week').agg({
            'earnings_amount': 'sum',
            'expenses_amount': 'sum'
        }).reset_index()
        x_axis = 'week'
    elif period == "Monthly":
        ts_data = filtered_df.groupby('month').agg({
            'earnings_amount': 'sum',
            'expenses_amount': 'sum'
        }).reset_index()
        x_axis = 'month'
    elif period == "Quarterly":
        ts_data = filtered_df.groupby(['year', 'quarter']).agg({
            'earnings_amount': 'sum',
            'expenses_amount': 'sum'
        }).reset_index()
        ts_data['period'] = ts_data['year'].astype(str) + ' Q' + ts_data['quarter'].astype(str)
        x_axis = 'period'
    else:  # Yearly
        ts_data = filtered_df.groupby('year').agg({
            'earnings_amount': 'sum',
            'expenses_amount': 'sum'
        }).reset_index()
        x_axis = 'year'
    
    # Plot trend
    fig = px.line(ts_data, x=x_axis, y=['earnings_amount', 'expenses_amount'],
                 title=f"{period} Trend Analysis", labels={'value': 'Amount (৳)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Decomposition analysis
    if period == "Monthly" and len(ts_data) >= 24:  # Need at least 2 years for decomposition
        st.subheader("Seasonal Decomposition")
        
        # Prepare time series
        ts_data['date'] = pd.to_datetime(ts_data['month'])
        ts_data = ts_data.set_index('date').sort_index()
        
        # Earnings decomposition
        st.markdown("### Earnings Decomposition")
        earnings_series = ts_data['earnings_amount']
        result = seasonal_decompose(earnings_series, model='additive', period=12)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
        result.observed.plot(ax=ax1, title='Observed')
        result.trend.plot(ax=ax2, title='Trend')
        result.seasonal.plot(ax=ax3, title='Seasonal')
        result.resid.plot(ax=ax4, title='Residual')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Expenses decomposition
        st.markdown("### Expenses Decomposition")
        expenses_series = ts_data['expenses_amount']
        result = seasonal_decompose(expenses_series, model='additive', period=12)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
        result.observed.plot(ax=ax1, title='Observed')
        result.trend.plot(ax=ax2, title='Trend')
        result.seasonal.plot(ax=ax3, title='Seasonal')
        result.resid.plot(ax=ax4, title='Residual')
        plt.tight_layout()
        st.pyplot(fig)

def forecasting_page(filtered_df):  # Add parameter here
    st.title("🔮 Financial Forecasting")
    
    if filtered_df.empty or 'date' not in filtered_df.columns:
        st.warning("Not enough data for forecasting")
        return
    
    # Forecasting parameters
    periods = st.slider("Forecast Period (months)", 1, 12, 6, key="forecast_period_slider")
    
    # Generate forecast
    forecast_data = forecast_finances(filtered_df, periods)
    
    if forecast_data is None:
        return
    
    # Plot forecast
    fig = px.line(forecast_data, 
                 x='date', 
                 y=['earnings_amount', 'expenses_amount', 'invesment_amount'],
                 color='type',
                 title=f"{periods}-Month Financial Forecast",
                 labels={'value': 'Amount (৳)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast summary
    st.subheader("Forecast Summary")
    
    last_actual = forecast_data[forecast_data['type'] == 'actual'].iloc[-1]
    forecast_period = forecast_data[forecast_data['type'] == 'forecast']
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate percentage changes safely
    def safe_percent_change(new, old):
        return ((new - old) / old * 100) if old != 0 else 0
    
    col1.metric("Projected Earnings", 
               f"৳{forecast_period['earnings_amount'].sum():,.2f}",
               delta=f"{safe_percent_change(forecast_period['earnings_amount'].sum(), last_actual['earnings_amount']):.1f}%")
    
    col2.metric("Projected Expenses", 
               f"৳{forecast_period['expenses_amount'].sum():,.2f}",
               delta=f"{safe_percent_change(forecast_period['expenses_amount'].sum(), last_actual['expenses_amount']):.1f}%")
    
    col3.metric("Projected Investment", 
               f"৳{forecast_period['invesment_amount'].sum():,.2f}",
               delta=f"{safe_percent_change(forecast_period['invesment_amount'].sum(), last_actual['invesment_amount']):.1f}%")
    
    # Show forecast data
    st.subheader("Forecast Data")
    st.dataframe(forecast_data.sort_values('date', ascending=False), use_container_width=True)


def budget_planning_page(filtered_df):  # Add parameter here
    st.title("💰 Budget Planning")
    
    if filtered_df.empty:
        st.warning("No data available for budget planning")
        return
    
    # Historical spending analysis
    st.subheader("Historical Spending Analysis")
    
    # Get average monthly spending by category
    monthly_cat = filtered_df.groupby(['month', 'main_category'])['expenses_amount'].sum().unstack().fillna(0)
    avg_monthly = monthly_cat.mean().sort_values(ascending=False)
    
    fig = px.bar(avg_monthly, 
                x=avg_monthly.index, 
                y=avg_monthly.values,
                title="Average Monthly Spending by Category",
                labels={'x': 'Category', 'y': 'Amount (৳)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Budget allocation
    st.subheader("Budget Allocation Plan")
    
    with st.form("budget_form", clear_on_submit=True):
        # Get available months from data
        available_months = pd.to_datetime(filtered_df['month']).dt.strftime("%B %Y").unique()
        budget_month = st.selectbox(
            "Plan for Month", 
            available_months,
            key="budget_month_select"
        )
        
        # Create budget inputs based on historical categories
        budget_items = {}
        cols = st.columns(3)
        
        for i, (category, amount) in enumerate(avg_monthly.items()):
            with cols[i % 3]:
                budget_items[category] = st.number_input(
                    f"{category} Budget",
                    min_value=0.0,
                    value=float(amount),
                    step=100.0,
                    key=f"budget_{category}"
                )
        
        submitted = st.form_submit_button("Save Budget Plan")
        
        if submitted:
            # Calculate totals
            total_budget = sum(budget_items.values())
            st.success(f"Budget plan for {budget_month} saved!")
            
            # Display budget summary
            st.subheader("Budget Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Budget", f"৳{total_budget:,.2f}")
                
            with col2:
                st.metric("Average Spending", f"৳{avg_monthly.sum():,.2f}")
            
            # Show budget allocation
            st.subheader("Budget Allocation")
            budget_df = pd.DataFrame({
                'Category': budget_items.keys(),
                'Budget Amount': budget_items.values(),
                'Average Spending': avg_monthly.values
            })
            st.dataframe(budget_df, use_container_width=True)

def performance_metrics_page(filtered_df):  # Add parameter here
    st.title("📉 Performance Metrics")
    
    if filtered_df.empty:
        st.warning("No data available for performance metrics")
        return
    
    # Calculate metrics using the filtered data
    metrics = calculate_financial_health(filtered_df)
    
    # Financial ratios section
    st.subheader("Financial Ratios")
    ratio_col1, ratio_col2, ratio_col3 = st.columns(3)
    
    with ratio_col1:
        st.metric("Savings Rate", 
                 f"{metrics.get('savings_rate', 0):.1f}%",
                 help="Net profit as percentage of earnings")
    
    with ratio_col2:
        st.metric("Expense Ratio", 
                 f"{metrics.get('expense_ratio', 0):.1f}%",
                 help="Expenses as percentage of earnings")
    
    with ratio_col3:
        st.metric("Investment Ratio", 
                 f"{metrics.get('investment_ratio', 0):.1f}%",
                 help="Investment as percentage of earnings")
    
    # Monthly averages section
    st.subheader("Monthly Averages")
    avg_col1, avg_col2, avg_col3 = st.columns(3)
    
    with avg_col1:
        st.metric("Avg Monthly Earnings", 
                 f"৳{metrics.get('avg_monthly_earnings', 0):,.2f}")
    
    with avg_col2:
        st.metric("Avg Monthly Expenses", 
                 f"৳{metrics.get('avg_monthly_expenses', 0):,.2f}")
    
    with avg_col3:
        st.metric("Avg Monthly Savings", 
                 f"৳{metrics.get('avg_monthly_savings', 0):,.2f}",
                 delta_color="inverse" if metrics.get('avg_monthly_savings', 0) < 0 else "normal")
    
    # Spending efficiency by category
    if not filtered_df.empty and 'expense_type' in filtered_df.columns:
        st.subheader("Spending Efficiency by Category")
        
        # Calculate efficiency metrics
        cat_efficiency = filtered_df.groupby('expense_type').agg({
            'expenses_amount': 'sum',
            'earnings_amount': 'sum'
        })
        
        # Calculate efficiency ratio safely (avoid division by zero)
        cat_efficiency['efficiency'] = np.where(
            cat_efficiency['expenses_amount'] > 0,
            cat_efficiency['earnings_amount'] / cat_efficiency['expenses_amount'],
            0
        )
        
        # Create visualization
        fig = px.bar(
            cat_efficiency.reset_index(),
            x='expense_type',
            y='efficiency',
            title="৳ Earned per ৳ Spent by Category",
            labels={'efficiency': 'Efficiency Ratio', 'expense_type': 'Category'}
        )
        fig.update_layout(yaxis_tickformat=".2f")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display raw data
        with st.expander("View Detailed Efficiency Data"):
            st.dataframe(
                cat_efficiency.sort_values('efficiency', ascending=False),
                use_container_width=True
            )

def data_management_page(filtered_df=None):  # Make parameter optional
    st.title("✏️ Data Management")
    
    # Use session state to manage the main dataframe
    if 'df' not in st.session_state:
        st.session_state.df = load_data(path)
    
    tab1, tab2, tab3 = st.tabs(["Edit Data", "Backup/Restore", "Data Quality"])
    
    with tab1:
        st.subheader("Edit Existing Data")
        
        if st.session_state.df.empty:
            st.warning("No data available to edit")
        else:
            edited_df = st.data_editor(
                st.session_state.df,
                use_container_width=True,
                num_rows="dynamic",
                key="data_editor"
            )
            
            if st.button("Save Changes", key="save_changes_btn"):
                try:
                    st.session_state.df = edited_df
                    if save_data(st.session_state.df, path):
                        st.success("Changes saved successfully!")
                    else:
                        st.error("Failed to save changes")
                except Exception as e:
                    st.error(f"Error saving changes: {str(e)}")
    
    with tab2:
        st.subheader("Backup & Restore")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Backup Data")
            if st.button("Create Backup", key="create_backup_btn"):
                backup_name = f"financial_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                if save_data(st.session_state.df, backup_name):
                    st.success(f"Backup created: {backup_name}")
                else:
                    st.error("Backup failed")
            
            # List available backups
            backups = [f for f in os.listdir() if f.startswith('financial_data_backup_') and f.endswith('.xlsx')]
            if backups:
                st.markdown("### Available Backups")
                selected_backup = st.selectbox(
                    "Select backup to restore", 
                    backups,
                    key="backup_select"
                )
                
                if st.button("Restore Selected Backup", key="restore_backup_btn"):
                    try:
                        backup_df = pd.read_excel(selected_backup)
                        st.session_state.df = backup_df
                        if save_data(st.session_state.df, path):
                            st.success("Backup restored successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to save restored data")
                    except Exception as e:
                        st.error(f"Restore failed: {str(e)}")
            else:
                st.info("No backups available")
        
        with col2:
            st.markdown("### Export/Import")
            
            # Export current data
            if not st.session_state.df.empty:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.df.to_excel(writer, index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="Download Current Data",
                    data=excel_data,
                    file_name="financial_data_export.xlsx",
                    mime="application/vnd.ms-excel",
                    key="export_data_btn"
                )
            
            # Import new data
            uploaded_file = st.file_uploader(
                "Upload Excel File", 
                type=['xlsx'],
                key="data_uploader"
            )
            if uploaded_file is not None:
                try:
                    new_df = pd.read_excel(uploaded_file)
                    st.success("File uploaded successfully!")
                    
                    if st.button("Replace Current Data", key="replace_data_btn"):
                        st.session_state.df = new_df
                        if save_data(st.session_state.df, path):
                            st.success("Data replaced successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to save new data")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    with tab3:
        st.subheader("Data Quality Check")
        
        if st.session_state.df.empty:
            st.warning("No data to analyze")
        else:
            # Missing values check
            st.markdown("### Missing Values")
            missing = st.session_state.df.isnull().sum()
            fig = px.bar(
                missing, 
                x=missing.index, 
                y=missing.values,
                title="Missing Values by Column",
                labels={'x': 'Column', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data consistency checks
            st.markdown("### Data Consistency")
            
            # Check for negative amounts where they shouldn't be
            negative_col1, negative_col2 = st.columns(2)
            
            with negative_col1:
                negative_earnings = (st.session_state.df['earnings_amount'] < 0).sum()
                st.metric("Negative Earnings Entries", negative_earnings)
            
            with negative_col2:
                negative_expenses = (st.session_state.df['expenses_amount'] < 0).sum()
                st.metric("Negative Expenses Entries", negative_expenses)
            
            if negative_earnings > 0 or negative_expenses > 0:
                st.warning("Negative values found in amounts. Check data integrity.")
            
            # Data type validation
            st.markdown("### Data Types")
            st.dataframe(
                pd.DataFrame(st.session_state.df.dtypes, columns=['Data Type']),
                use_container_width=True
            )


# Add to the bottom of your script (after all page functions but before the main execution)

def managing_director_page(filtered_df):
    # Set up the page with a title and introduction
    st.title("Managing Director Dashboard")
    st.write(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    st.write("To: The Managing Director")
    st.write("Welburg Metal Pvt Ltd.")
    st.write("\nSubject: Monthly Financial Summary\n")
    
    st.write("Dear Sir,")
    st.write("Please find below the key financial metrics for your review. This report highlights our performance across critical categories:")

    # Date range selector at the top
    st.subheader("Select Date Range")
    min_date = filtered_df['date'].min() if not filtered_df.empty else date.today()
    max_date = filtered_df['date'].max() if not filtered_df.empty else date.today()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("To date", max_date, min_value=min_date, max_value=max_date)
    
    # Filter data based on selected date range
    date_filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                                 (filtered_df['date'] <= end_date)]

    # Calculate metrics for the selected period
    total_earnings = date_filtered_df['earnings_amount'].sum()
    total_expenses = date_filtered_df['expenses_amount'].sum()
    net_profit = total_earnings - total_expenses

    # 1. Monthly Performance Bar Chart
    st.subheader("📅 Monthly Financial Performance")
    if not date_filtered_df.empty and 'month' in date_filtered_df.columns:
        monthly_data = date_filtered_df.groupby('month').agg({
            'earnings_amount': 'sum',
            'expenses_amount': 'sum',
            'invesment_amount': 'sum'
        }).reset_index()
        
        fig = px.bar(monthly_data, 
                    x='month',
                    y=['earnings_amount', 'expenses_amount', 'invesment_amount'],
                    barmode='group',
                    title=f"Monthly Summary ({start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')})",
                    labels={'value': 'Amount (৳)', 'variable': 'Category'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No monthly data available for selected period")

    # 2. Category-wise Expense Breakdown
    st.subheader("🗂 Category-wise Expenditure")
    if not date_filtered_df.empty and 'main_category' in date_filtered_df.columns:
        category_data = date_filtered_df.groupby('main_category')['expenses_amount'].sum().reset_index()
        
        fig = px.bar(category_data,
                    x='main_category',
                    y='expenses_amount',
                    color='main_category',
                    title=f"Expenses by Main Category (৳)",
                    labels={'expenses_amount': 'Amount (৳)', 'main_category': 'Category'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top 3 categories
        top_categories = category_data.nlargest(3, 'expenses_amount')
        st.write("Top 3 Expense Categories:")
        for idx, row in top_categories.iterrows():
            st.write(f"- {row['main_category']}: ৳{row['expenses_amount']:,.2f}")
    else:
        st.warning("No category data available for selected period")

    # 3. Expense Type Analysis
    st.subheader("🧾 Expense Type Distribution")
    if not date_filtered_df.empty and 'expense_type' in date_filtered_df.columns:
        type_data = date_filtered_df.groupby('expense_type')['expenses_amount'].sum().reset_index()
        
        fig = px.bar(type_data,
                    x='expense_type',
                    y='expenses_amount',
                    color='expense_type',
                    title=f"Expenses by Type (৳)",
                    labels={'expenses_amount': 'Amount (৳)', 'expense_type': 'Type'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No expense type data available for selected period")

    # Key Metrics Summary
    st.subheader("🔍 Key Metrics Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Earnings", f"৳{total_earnings:,.2f}")
    col2.metric("Total Expenses", f"৳{total_expenses:,.2f}")
    col3.metric("Net Profit", f"৳{net_profit:,.2f}", 
               delta_color="inverse" if net_profit < 0 else "normal")

    # Additional Data Request Section
    st.subheader("Request Additional Information")
    with st.expander("Click here to request additional data"):
        additional_request = st.text_area("Please specify what additional data or analysis you would like to receive:")
        
        if st.button("Submit Request"):
            if additional_request:
                st.success("Thank you! Your request has been submitted to the Accounts Department.")
                # Here you would add code to actually store/send the request
            else:
                st.warning("Please describe what additional information you need")

    # Closing signature
    st.write("\n\nThank you for your time and consideration.\n")
    st.write("Sincerely,")
    st.write("Mujakkir Ahmad")
    st.write("Accounts Department")
    st.write("Welburg Metal Pvt Ltd.")
    st.write("Contact: mujakkirar4@gmail.com | Phone: +880 1958 385955")


def footer():
    st.markdown("---")
    st.caption(f"""
    <div style="text-align: center;">
        <p>© {datetime.now().year} Financial Analytics Dashboard • Developed by :<strong> Mujakkir Ahmad | Data Analyst </strong></p>
        <p>V1.0.0 • Powered by Python, Streamlit, and Plotly</p>
        <p style="font-size: 0.8em;">
            <a href="https://github.com/mujakkirdv/" target="_blank">GitHub</a> • 
            <a href="https://webmujakkir.streamlit.app/" target="_blank">Portfolio</a> • 
            <a href="mailto:mujakkirar4@gmail.com">Contact Support</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# Then modify your main execution block:
if __name__ == "__main__":
    sidebar_navigation()
    footer()


st.markdown("---")
st.caption("Powerd by : WELBURG METAL PVT LTD | ACCOUNT DEPARTMENT")

    
