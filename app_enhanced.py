from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration with a custom theme
st.set_page_config(
    page_title="Smart Grid Load Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Containers */
    .st-emotion-cache-1y4p8pa {
        background-color: #1a1f2c !important;
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #2d3958;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #00cf86 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #9ba3b0 !important;
    }
    
    /* Charts */
    .js-plotly-plot {
        background-color: #1a1f2c !important;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1f2c;
        border-right: 1px solid #2d3958;
    }
    
    /* Buttons and selectors */
    .stSelectbox > div > div {
        background-color: #2d3958 !important;
        color: #ffffff !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #2d3958 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    /* Markdown text */
    .st-ae {
        color: #ffffff !important;
    }
    
    /* Metric containers */
    .metric-card {
        background-color: #2d3958;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 1px solid #3d4b6e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Links */
    a {
        color: #00cf86 !important;
    }
    
    /* Dividers */
    hr {
        border-color: #2d3958 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(selected_year):
    files = {
        2023: 'utilities/datasets/2023_load_data_clean.csv',
        2024: 'utilities/datasets/load_data_2024_cleaned.csv',
        2025: 'utilities/datasets/load_data_2025_cleaned.csv'
    }
    data = pd.read_csv(files[selected_year])
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
    return data

def predict_tomorrow_load(data):
    """Predict tomorrow's load using historical averages and recent trends"""
    # Get the last 7 days of data
    last_week = data.tail(288 * 7)  # 288 intervals per day
    
    # Calculate average daily pattern
    last_week['time_of_day'] = last_week['datetime'].dt.time
    avg_pattern = last_week.groupby('time_of_day')['load'].mean()
    
    # Calculate trend
    daily_avg = last_week.groupby(last_week['datetime'].dt.date)['load'].mean()
    trend = (daily_avg.iloc[-1] - daily_avg.iloc[0]) / 7
    
    # Create tomorrow's datetime index
    tomorrow = pd.date_range(
        start=data['datetime'].max() + timedelta(days=1),
        periods=288,
        freq='5T'
    )
    
    # Generate predictions
    base_predictions = [avg_pattern[t.time()] for t in tomorrow]
    trend_adjusted = [p + trend for p in base_predictions]
    
    # Add some randomness to make it more realistic
    noise = np.random.normal(0, avg_pattern.std() * 0.1, 288)
    final_predictions = [max(0, p + n) for p, n in zip(trend_adjusted, noise)]
    
    return pd.DataFrame({
        'datetime': tomorrow,
        'predicted_load': final_predictions
    })

def generate_fallback_prediction(data):
    """Generate a fallback prediction based on historical averages"""
    # Calculate average load pattern for the last week
    last_week = data.tail(288 * 7)
    avg_pattern = last_week.groupby(last_week['datetime'].dt.time)['load'].mean()
    
    # Create tomorrow's datetime index
    tomorrow = pd.date_range(
        start=data['datetime'].max() + timedelta(days=1),
        periods=288,
        freq='5T'
    )
    
    # Create predictions using the average pattern
    predictions = [avg_pattern[t.time()] for t in tomorrow]
    
    return pd.DataFrame({
        'datetime': tomorrow,
        'predicted_load': predictions
    })

def custom_resampler(arraylike):
    return np.sum(arraylike) / (12 * 1000)

def show_header():
    st.title("⚡ Smart Grid Load Predictor")
    st.markdown("---")

def show_sidebar_filters():
    st.sidebar.header("📊 Data Filters")
    selected_year = st.sidebar.selectbox(
        "Select Year",
        [2023, 2024, 2025],
        format_func=lambda x: f"Year {x}"
    )
    return selected_year

def show_overview_metrics(data):
    st.header("📈 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Daily Load",
            f"{int(data['load'].mean()):,} MW",
            delta=f"{int(data['load'].mean() - data['load'].shift(288).mean()):,} MW"
        )
    
    with col2:
        st.metric(
            "Peak Load",
            f"{int(data['load'].max()):,} MW",
            delta=f"{int(data['load'].max() - data['load'].shift(288).max()):,} MW"
        )
    
    with col3:
        st.metric(
            "Total Energy",
            f"{int(data['load'].sum() / 12000):,} GWh",
            delta=f"{int((data['load'].sum() - data['load'].shift(288).sum()) / 12000):,} GWh"
        )
    
    with col4:
        efficiency = round((data['load'].mean() / data['load'].max()) * 100, 2)
        st.metric(
            "Load Factor",
            f"{efficiency}%",
            delta=f"{round(efficiency - 100, 2)}% from ideal"
        )

def show_load_prediction(data):
    st.header("🔮 Tomorrow's Load Prediction")
    
    # Get tomorrow's prediction
    tomorrow_pred = predict_tomorrow_load(data)
    
    # Create the plot using plotly
    fig = make_subplots(rows=1, cols=1)
    
    # Add actual load for today
    today_data = data.tail(288)
    fig.add_trace(
        go.Scatter(
            x=today_data['datetime'],
            y=today_data['load'],
            name="Today's Load",
            line=dict(color='#00cf86', width=2)
        )
    )
    
    # Add predicted load for tomorrow
    fig.add_trace(
        go.Scatter(
            x=tomorrow_pred['datetime'],
            y=tomorrow_pred['predicted_load'],
            name="Predicted Load",
            line=dict(color='#ff6b6b', width=2, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Load Prediction for Tomorrow",
        xaxis_title="Time",
        yaxis_title="Load (MW)",
        hovermode='x unified',
        showlegend=True,
        template="plotly_dark",
        plot_bgcolor='rgba(26,31,44,1)',
        paper_bgcolor='rgba(26,31,44,1)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show prediction metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Peak Load",
            f"{int(tomorrow_pred['predicted_load'].max()):,} MW",
            delta=f"{int(tomorrow_pred['predicted_load'].max() - today_data['load'].max()):,} MW vs Today"
        )
    
    with col2:
        st.metric(
            "Predicted Average Load",
            f"{int(tomorrow_pred['predicted_load'].mean()):,} MW",
            delta=f"{int(tomorrow_pred['predicted_load'].mean() - today_data['load'].mean()):,} MW vs Today"
        )
    
    with col3:
        predicted_energy = tomorrow_pred['predicted_load'].sum() / 12000
        today_energy = today_data['load'].sum() / 12000
        st.metric(
            "Predicted Energy Consumption",
            f"{predicted_energy:.2f} GWh",
            delta=f"{(predicted_energy - today_energy):.2f} GWh vs Today"
        )

def main():
    show_header()
    selected_year = show_sidebar_filters()
    
    # Load data
    data = load_data(selected_year)
    
    # Show metrics and visualizations
    show_overview_metrics(data)
    st.markdown("---")
    
    # Show prediction for tomorrow
    show_load_prediction(data)
    st.markdown("---")
    
    # Additional views can be added here
    
if __name__ == "__main__":
    main()
