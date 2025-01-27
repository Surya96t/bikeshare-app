import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_cleaned_data():
    """ Returns the cleaned data """
    
    data = pd.read_csv(r"data/SeoulBikeData_cleaned_cols.csv")
    
    return data 


def sidebar_input():
    """ Returns the sidebar input fields """
    # Create Streamlit input fields for each element in the dictionary in the sidebar
    with st.sidebar:
        st.write("Input your data:")
        hour = st.number_input('Hour', value=0)
        temp = st.number_input('Temperature (°C)', value=-5.0)
        humidity = st.number_input('Humidity (%)', value=60)
        wind_speed = st.number_input('Wind Speed (m/s)', value=2.0)
        visibility = st.number_input('Visibility (m)', value=2000)
        solar_rad = st.number_input('Solar Radiation (W/m²)', value=0.0)
        rainfall = st.number_input('Rainfall (mm)', value=0.0)
        snowfall = st.number_input('Snowfall (mm)', value=0.0)
        seasons = st.selectbox('Seasons', ['Winter', 'Spring', 'Summer', 'Autumn'])
        holiday = st.selectbox('Holiday', ['No Holiday', 'Holiday'])
        day = st.selectbox('Day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        month = st.selectbox('Month', ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        
    input_dict = {
        'hour': hour,
        'temp': temp,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'visibility': visibility,
        'solar_rad': solar_rad,
        'rainfall': rainfall,
        'snowfall': snowfall,
        'seasons': seasons,
        'holiday': holiday,
        'day': day,
        'month': month
    }
        
    return input_dict


def create_day_month_year_columns(bikeshare):
    bikeshare["date"] = pd.to_datetime(bikeshare["date"], dayfirst=True) # dayfirst=True is used to specify the date format

    # Extract day, month, and year
    bikeshare["day"] = bikeshare["date"].dt.day
    bikeshare["month"] = bikeshare["date"].dt.month
    bikeshare["year"] = bikeshare["date"].dt.year
    
    return bikeshare


def hourly_rentals_plot(day, month, year, data, avg_rentals=False):
    
    bikeshare = create_day_month_year_columns(data)
    
    avg_hourly_rentals = bikeshare.groupby('hour').agg(avg_rented_bike_count = ('rented_bike_count', 'mean'))
    
    data = bikeshare[(bikeshare["day"] == day) & (bikeshare["month"] == month) & (bikeshare["year"] == year)]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(x="hour", y="rented_bike_count", data=data)
    if avg_rentals is True:
        sns.lineplot(data=avg_hourly_rentals, x=avg_hourly_rentals.index, y='avg_rented_bike_count')
    # sns.lineplot(data=avg_hourly_rentals, x=avg_hourly_rentals.index, y='avg_rented_bike_count')
    plt.title(f"Hourly Bike Rentals on {day}-{month}-{year}")
    plt.xlabel("Hour")
    plt.ylabel("Rented Bike Count")
    plt.xticks(avg_hourly_rentals.index)
    
    
    st.pyplot(fig, use_container_width=True)
    
    
def get_monthly_rentals(month_num, year_num, bikeshare):
    """
    This function analyzes bike rental data for December 2017 from a bikeshare dataset.

    Steps performed:
    1. Filters the dataset for records from December 2017.
    2. Aggregates the total number of bike rentals for each day in December 2017.
    3. Adds a new column indicating the day of the week for each date.
    4. Calculates the average number of rentals for each day of the week (e.g., Monday, Tuesday).
    5. Categorizes each day as 'Weekday' or 'Weekend' based on the day of the week.
    6. Aggregates the average rentals by 'Weekday' and 'Weekend' categories.
    7. Merges the daily rental data with the average rentals by day of the week and by weekday/weekend category.

    Returns a DataFrame with detailed rental statistics for December 2017, including variations by date, day of the week, and day category (Weekday/Weekend).
    """
    
    month_rentals = bikeshare[(bikeshare['month'] == month_num) & (bikeshare['year'] == year_num)] # December 2017 data
    rentals = month_rentals.groupby(['date']).agg(total_rentals=('rented_bike_count', 'sum')).reset_index() # Total rentals in December 2017
    rentals['day_of_the_week'] = rentals['date'].dt.day_name() # Day of the week
    
    daily_month_rentals = rentals.groupby('day_of_the_week').agg(avg_rentals=('total_rentals', 'mean')).reset_index()
    daily_month_rentals['day_cat'] = daily_month_rentals['day_of_the_week'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')
    daily_month_rentals_cat = daily_month_rentals.groupby(['day_cat']).agg(rental_byCat=("avg_rentals", "sum")).reset_index()
    
    month_merge = pd.merge(rentals, daily_month_rentals, on="day_of_the_week", how="left")
    month_merge = pd.merge(month_merge, daily_month_rentals_cat, on="day_cat", how="left")
    
    return month_merge

def plot_monthly_rentals(df_month):
    """This function simply plots the total and average rentals by month.

    Args:
        df_month (DataFrame): This is the aggregated data for the month of interest.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.set_style('darkgrid')


    sns.lineplot(data=df_month, x="date", y="total_rentals", label="Total Rentals")
    sns.lineplot(data=df_month, x="date", y="avg_rentals", label="Average Rentals of the Month")

    plt.title(f"Total vs Average Rentals by Month ({df_month['date'].dt.month_name().unique()[0]})")
    plt.ylabel("Total Rentals")
    plt.xlabel("Date")
    plt.xticks(df_month['date'], rotation=90)
    plt.legend()
    
    
    st.pyplot(fig, use_container_width=True)
    
def monthly_rentals(month, year, data):
    df = get_monthly_rentals(month, year, data)
    plot_monthly_rentals(df)
    
    return df
    
    
def rental_plot_by_week(merge_bikeshare):
    color = sns.color_palette("Paired")
    
    days_of_week = merge_bikeshare.groupby('day_of_the_week')['avg_rentals'].mean().reset_index()    
    days_of_week.set_index('day_of_the_week', inplace=True)

    new_index = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    days_of_week = days_of_week.reindex(new_index)
    
    fig, ax = plt.subplots(1,2, figsize=(15, 5))

    plt.suptitle('Bike Rentals based on Day of Week')
    day_bars = sns.barplot(x=days_of_week.index, y=days_of_week['avg_rentals'], ax=ax[0], palette=color)
    for i in range(len(day_bars.containers)):
        day_bars.bar_label(day_bars.containers[i])
    ax[0].set_title('Bike Rentals based on Day of Week for Dec 2017')
    ax[0].set_xlabel('Day of Week')
    ax[0].set_ylabel('Average Rentals')


    plt.pie(x=days_of_week['avg_rentals'], labels=days_of_week.index, pctdistance=0.8, autopct='%1.2f%%', explode=(0,0,0.1,0,0,0,0), colors=color)
    centre_circle = plt.Circle((0,0),0.60,fc='white')
    p = plt.gcf()
    p.gca().add_artist(centre_circle)
    plt.text(0, 0, days_of_week['avg_rentals'].unique().sum(), ha='center', va='center', fontsize=16)
    plt.title('Proportion of Total Rentals for Dec 2017')

    plt.tight_layout()
    
    st.pyplot(fig, use_container_width=True)
    
    
def day_hour_melted_pivot_df(bikeshare):
    day_hour = bikeshare.groupby([bikeshare['date'].dt.day_name(), 'hour']).agg(total_rentals=('rented_bike_count', 'sum'))
    day_hour_pivot = pd.pivot_table(day_hour, index='date', columns='hour', values='total_rentals')
    day_hour_pivot.index.name = 'Day of Week'
    
    for i in range(0, 24):
        day_hour_pivot[i] = day_hour_pivot[i].astype(int)
    
    day_hour_pivot['Total'] = day_hour_pivot.sum(axis=1)
    
    new_index = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    day_hour_pivot = day_hour_pivot.reindex(new_index) # Reorder the rows based on the day of the week
    
    day_hour_melted = day_hour_pivot.reset_index().melt(id_vars='Day of Week', value_vars=day_hour_pivot.columns[:-1], var_name='hour', value_name='total_rentals')
    
    return day_hour_pivot, day_hour_melted

def plot_day_hour_rentals(day_hour_pivot, day_hour_melted):
    
    color = sns.color_palette("Paired")
    
    fig, ax = plt.subplots(2, 1, figsize=(22, 14))

    # plt.subplot(2, 1, 1)
    sns.lineplot(data=day_hour_melted, x='hour', y='total_rentals', hue='Day of Week', palette=color, ax=ax[0])
    plt.title('Total Rentals per hour', fontsize=14)
    plt.xlabel('Time')
    plt.xticks(np.arange(0,24,1))
    plt.ylabel('Total')

    # plt.subplot(2, 1, 2)
    sns.heatmap(day_hour_pivot.iloc[:,:-1], cmap="coolwarm", annot=True, fmt="d", linewidths=0.5, ax=ax[1]) 
    plt.title('Total Trips by Day and Hour', fontsize=14)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Day of the Week')

    # plt.tight_layout()
    
    st.pyplot(fig, use_container_width=True)
    
    
def weather_impact_on_rental(bikeshare, weather_col):
    
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.regplot(data=bikeshare, x=weather_col, y='rented_bike_count', ax=ax) 
    st.pyplot(fig, use_container_width=True)
    
    
def seasonal_plot(bikeshare):
    season_rentals = bikeshare.groupby('seasons').agg(total = ('rented_bike_count', 'sum'))
    # season_rentals
    color = sns.color_palette("Paired") 
    
    fig, ax = plt.subplots(1,2, figsize=(15, 5))

    plt.suptitle('Seasonal Bike Rentals')
    day_bars = sns.barplot(x=season_rentals.index, y=season_rentals['total'], ax=ax[0], palette=color, ci=None)
    for i in range(len(day_bars.containers)):
        day_bars.bar_label(day_bars.containers[i], fmt='%d')
    ax[0].set_title('Bike Rentals based on Season')
    ax[0].set_xlabel('Season')
    ax[0].set_ylabel('Total Rentals')


    plt.pie(x=season_rentals['total'], labels=season_rentals.index, pctdistance=0.8, autopct='%1.2f%%', explode=(0,0,0.1,0), colors=color)
    centre_circle = plt.Circle((0,0),0.60,fc='white')
    p = plt.gcf()
    p.gca().add_artist(centre_circle)
    plt.text(0, 0, season_rentals['total'].unique().sum(), ha='center', va='center', fontsize=16)
    plt.title('Total Rentals')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)