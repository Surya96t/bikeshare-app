import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.dashboard_utils import get_bikeshare_data, preprocess_data, total_month_rentals_df, get_weekly_rentals_df, get_average_hourly_rentals_df, get_monthly_rentals_df, rentals_per_day, get_avg_temp_humidity


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
        #initial_sidebar_state="expanded",
        menu_items={
            "About": "https://github.com/Surya96t/bikeshare-app"
        }
    )
    
    # Load configuration and data
    config = Config.from_json(CFGLog)
    # data = get_cleaned_data()
    
    # data, filtered_data = preprocess_data(raw_data)
    #input_df = pd.DataFrame([input_data])
    inferrer = Inferrer()
    
    st.header("Bikeshare Rental Analytics", divider="blue")
    st.markdown("""
    Gain insights into bikeshare patterns through interactive visualizations and historical data analysis.
    Explore temporal trends, weather impacts, and seasonal variations in bike rental demand.
    """)

    with st.expander("See Raw Data"):
        raw_data = get_bikeshare_data()
        st.dataframe(raw_data)
    
    # Key Metrics
    #####################
    
    # Total Rentals
    
    df, filtered_df = preprocess_data(raw_data)
    total_rentals = int(df["rented_bike_count"].sum())
    
    ## Average Monthly Rentals
    monthly_avg = df.groupby('month')['rented_bike_count'].sum().reset_index()
    monthly_mean = monthly_avg["rented_bike_count"].mean()
    
    ## Peak Hour
    peak_hour = df.groupby('hour')['rented_bike_count'].sum().idxmax()
    min_hour = int(df.groupby('hour')['rented_bike_count'].sum().idxmin())
    
    with st.container():
        st.write("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rentals", value=total_rentals, border=True, delta=2.0, delta_color="inverse")
    
        with col2:
            st.metric("Average Monthly Rentals", value=monthly_mean, border=True, delta=2.0, delta_color="inverse")

        with col3:
            st.metric("Peak Hour", value=peak_hour, border=True, delta=min_hour, delta_color="inverse")
            
        with col4:
            with st.expander("Show More"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Check out GitHub Repo for entire analysis and source code.")
                with col2:
                    st.html(
                        "<a href='https://github.com/Surya96t' style='text-decoration: none; color: white; background-color: blue; padding: 2px 5px; border-radius: 5px; text-align: center; display: inline-block;'>GitHub Link</a>"
                    )
    
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            tab1, tab2, tab3, tab4 = st.tabs(["Overall Plot", "Monthly Plot", "Weekly Plot", "Hourly Plot"])
            
            with tab1:
                overall_plot = px.line(filtered_df, x=filtered_df.index, y="rented_bike_count", labels={"rented_bike_count": "Rented Bike Count", "index": "Time"}, 
                title="Bike Rentals Over Time")
                st.plotly_chart(overall_plot)
            
            with tab2:
                overall_months, single_month = st.tabs(["All Months", "Particular Month"])
                with overall_months:
                    monthly_rentals_bar = total_month_rentals_df(df)
                    monthly_plot_bar = px.bar(monthly_rentals_bar, x="month", y="rented_bike_count", labels={"rented_bike_count": "Rented Bike Count", "month": "Month"},
                    title="Monthly Bike Rentals")
                    st.plotly_chart(monthly_plot_bar)
                
                with single_month: 
                    month_col , year_col = st.columns(2)
                    with month_col:
                        month_names = [
                            "January", "February", "March", "April", "May", "June", 
                            "July", "August", "September", "October", "November", "December"
                        ]
                        month = st.selectbox("Select Month", month_names)
                        month_num = month_names.index(month) + 1
                        
                    with year_col:
                        year = st.selectbox("Select Year", range(2017, 2019))
                    
                    st.info('Dates ranges between Dec-2017 and Nov-2018', icon="ℹ️")
                    monthly_rentals_line = get_monthly_rentals_df(df, month_num, year)
                    monthly_plot_line = px.line(
                        monthly_rentals_line,
                        x="date",
                        y=["rented_bike_count","avg_rentals"],
                        labels={"date": "Date", "rented_bike_count": "Rented Bike Count"},
                        title=f"Monthly Bike Rentals ({month} - {year})",
                        color_discrete_map={"rented_bike_count": "blue", "avg_rentals": "red"})
                    st.plotly_chart(monthly_plot_line)
                
                
            with tab3:
                weekly_rentals = get_weekly_rentals_df(df)
                weekly_plot = px.bar(weekly_rentals, x="year_week", y="total_rentals",
                labels={"year_week": "Week", "total_rentals": "Total Rentals"},
                title="Weekly Bike Rentals (Dec 2017 - Nov 2018)")
                st.plotly_chart(weekly_plot)
                
            with tab4:
                overall_days, single_day = st.tabs(["Mean All Hours", "Particular Day"])
                
                with overall_days:
                    hourly_rentals = get_average_hourly_rentals_df(df)
                    hourly_plot = px.line(hourly_rentals, x="hour", y="rented_bike_count", labels={"hour": "Hour", "rented_bike_count": "Rented Bike Count"})
                    st.plotly_chart(hourly_plot)
                    
                    
                with single_day:
                    d = st.date_input(
                        "Select Month and Year",
                        datetime.date(2017, 12, 1),
                        min_value=datetime.date(2017, 1, 1),
                        max_value=datetime.date(2019, 1, 1)
                    )
                    st.info('Dates ranges between Dec-2017 and Nov-2018 only and there are days when the serivce is closed. ', icon="ℹ️")
                    per_hour_data = rentals_per_day(df, d.day, d.month, d.year)
                    per_hour_plot = px.line(per_hour_data, x="hour", y="rented_bike_count", labels={"hour": "Hour", "rented_bike_count": "Rented Bike Count"})
                    st.plotly_chart(per_hour_plot)
                    


        with col2:
            seasonal_tab, temp_tab, humidity_tab = st.tabs(["Seasonal Rentals", "Average Temp", "Average Humidity"])
            
            with seasonal_tab:
                season_rentals = raw_data.groupby('seasons').agg(total=('rented_bike_count', 'sum'))
                
                fig, ax = plt.subplots(figsize=(15, 5))
                
                # Set background color to black
                fig.patch.set_facecolor('#0E1117')
                ax.set_facecolor('#0E1117')

                color = sns.color_palette("dark")

                wedges, texts, autotexts = plt.pie(
                    x=season_rentals['total'],
                    labels=season_rentals.index,
                    pctdistance=0.8,
                    autopct='%1.2f%%',
                    explode=(0, 0, 0.1, 0),
                    colors=color,
                    textprops={'color': 'white'}  # Set text color to white
                )

                centre_circle = plt.Circle((0, 0), 0.60, fc='#0E1117')  # Change center to black
                p = plt.gcf()
                p.gca().add_artist(centre_circle)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            
            with temp_tab:
                avg_temp_hum_data = get_avg_temp_humidity()
                st.dataframe(avg_temp_hum_data[[ "avg_temperature"]], use_container_width=True)
                
                
            with humidity_tab:
                st.dataframe(avg_temp_hum_data[["avg_humidity"]], use_container_width=True)





if __name__ == "__main__":
    main()
