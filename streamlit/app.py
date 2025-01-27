import streamlit as st
import pandas as pd
import datetime
from utils.supportFunc import (
    get_cleaned_data, sidebar_input, hourly_rentals_plot, 
    monthly_rentals, rental_plot_by_week, day_hour_melted_pivot_df,
    plot_day_hour_rentals, weather_impact_on_rental, seasonal_plot
)
from bikeshare.utils.config import Config
from bikeshare.configs.config import CFGLog
from bikeshare.executor.inferrer import Inferrer  
import warnings
warnings.filterwarnings('ignore')

def main():
    # Set wide page configuration
    st.set_page_config(
        page_title="Bikeshare Analytics Dashboard",
        page_icon="🚲",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "https://github.com/Surya96t/bikeshare-app"
        }
    )

    # Custom CSS styling
    st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        h1 {color: #2c3e50;}
        h2 {color: #34495e; border-bottom: 2px solid #3498db;}
        .stDateInput, .stSelectbox {max-width: 300px;}
        .stTabs [data-baseweb="tab-list"] {gap: 10px;}
        .stTabs [data-baseweb="tab"] {padding: 8px 20px; border-radius: 4px;}
        .stDataFrame {border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

    # Load configuration and data
    config = Config.from_json(CFGLog)
    data = get_cleaned_data()
    input_data = sidebar_input()
    input_df = pd.DataFrame([input_data])
    inferrer = Inferrer()

    # Define main tabs
    home_tab, xgb_tab = st.tabs(["📊 Analytics Dashboard", "🤖 Predictive Model"])

    with home_tab:
        # Header Section
        st.header("Bikeshare Rental Analytics", divider="blue")
        st.markdown("""
        Gain insights into bikeshare patterns through interactive visualizations and historical data analysis.
        Explore temporal trends, weather impacts, and seasonal variations in bike rental demand.
        """)

        # Dataset Overview
        with st.container():
            st.subheader("📁 Dataset Preview")
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Total Records", data.shape[0])
                st.metric("Features Available", data.shape[1])
            with c2:
                with st.expander("View Raw Data Sample", expanded=False):
                    st.dataframe(data.head(), use_container_width=True)
                    st.caption("First 5 rows of the processed dataset")

        # Analytics Sections
        st.divider()
        st.subheader("📈 Temporal Analysis")
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("### Daily Rentals Analysis")
            selected_date = st.date_input(
                "Select analysis date:",
                value=datetime.date(2017, 12, 1),
                min_value=datetime.date(2017, 12, 1),
                max_value=datetime.date(2018, 11, 30)
            )
            hourly_rentals_plot(selected_date.day, selected_date.month, selected_date.year, data, avg_rentals=True)

        with col2:
            st.markdown("### Monthly Trends")
            merge_df = monthly_rentals(selected_date.month, selected_date.year, data)
            rental_plot_by_week(merge_df)

        st.divider()
        st.subheader("⏰ Hourly Patterns")
        pivot, melted = day_hour_melted_pivot_df(data)
        plot_day_hour_rentals(pivot, melted)

        st.divider()
        st.subheader("🌤 Environmental Factors")
        
        env_col1, env_col2 = st.columns(2, gap="large")
        with env_col1:
            weather_feature = st.selectbox(
                "Select weather parameter:",
                data.columns[3:11],
                index=0,
                key="weather_select"
            )
            weather_impact_on_rental(data, weather_feature)

        with env_col2:
            st.markdown("### Seasonal Impact Analysis")
            seasonal_plot(data)

    with xgb_tab:
        st.header("XGBoost Demand Forecasting", divider="blue")
        st.markdown("Real-time predictions using our optimized gradient boosting model")
        
        # Model Information
        xgb_metrics_path = config.output.output_path + "XGBoost_metrics.json"
        xgb_metrics = pd.read_json(xgb_metrics_path)
        xgb_model_parameters = config.gradient_boosting
        
        # Prediction Interface
        with st.container():
            st.subheader("Live Prediction Interface")
            pred_col1, pred_col2, pred_col3 = st.columns([1,1,2], gap="medium")
            
            with pred_col1:
                st.markdown("### Input Parameters")
                st.json(input_data)
                if st.button("🔄 Generate Prediction", type="primary"):
                    prediction = inferrer.xgb_infer(input_df)
                    st.session_state.prediction = prediction
            
            with pred_col2:
                st.markdown("### Prediction Result")
                if 'prediction' in st.session_state:
                    st.metric(
                        label="Predicted Rentals",
                        value=st.session_state.prediction[0],
                        help="Estimated bikes needed based on current inputs"
                    )
                else:
                    st.info("Click 'Generate Prediction' to view results")
            
            with pred_col3:
                st.markdown("### Model Details")
                tab1, tab2 = st.tabs(["Performance Metrics", "Configuration"])
                with tab1:
                    st.dataframe(
                        xgb_metrics.T.style.format("{:.3f}"),
                        use_container_width=True
                    )
                with tab2:
                    st.json({
                        'n_estimators': xgb_model_parameters.n_estimators,
                        'max_depth': xgb_model_parameters.max_depth,
                        'subsample': xgb_model_parameters.subsample,
                        'learning_rate': xgb_model_parameters.learning_rate
                    })

if __name__ == "__main__":
    main()