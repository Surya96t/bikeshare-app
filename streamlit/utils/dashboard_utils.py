import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_bikeshare_data():
    
    # Load data
    data = pd.read_csv(r"data/SeoulBikeData_cleaned_cols.csv")
    
    return data


def preprocess_data(data):
    # Convert date column to datetime format
    data["date"] = pd.to_datetime(data["date"], dayfirst=True)
    
    # Extract day, month, and year
    data["day"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["year"] = data["date"].dt.year
    
    # Create datetime column
    data['datetime'] = data['date'] + pd.to_timedelta(data['hour'], unit='h')
    
    # Create daily aggregated data
    df = data[['datetime', 'rented_bike_count']].copy()
    df.set_index('datetime', inplace=True)
    daily_counts = df.resample('D').sum()
    filtered_df = daily_counts[daily_counts['rented_bike_count'] != 0]
    
    return data, filtered_df


def total_month_rentals_df(data):
    
    per_month_data = data.copy()
    # Group by month and sum rentals
    per_month_data = per_month_data.groupby('month')['rented_bike_count'].sum().reset_index()
    
    # Convert month number to month name for better visualization
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    per_month_data['month'] = per_month_data['month'].map(month_map)  
    
    return per_month_data



def get_monthly_rentals_df(data , month_num, year):
    
    month_df = data.copy()
    
    selected_month = month_df[(month_df['month'] == month_num) & (month_df['year'] == year)]
    monthly_rentals = selected_month.groupby('date')['rented_bike_count'].sum().reset_index()
    monthly_rentals['day_of_week'] = monthly_rentals['date'].dt.day_name()
    
    # Calculate average rentals per week
    average_per_week = monthly_rentals.groupby('day_of_week').agg(avg_rentals=('rented_bike_count', 'mean')).reset_index()
    
    # combine the two dataframes
    merged_df = pd.merge(monthly_rentals, average_per_week, on='day_of_week')
    
    return merged_df



def get_weekly_rentals_df(data):
    
    week_data = data.copy()
    # Group by month and sum rentals
    
    # week_data["week"] = week_data["data"].dt.isoclaendar().week
    week_data["year_week"] = week_data["date"].dt.strftime('%Y-W%V')  # Create unique "year-week" label
    
    weekly_rentals = week_data.groupby(['year_week']).agg(
        total_rentals=('rented_bike_count', 'sum')
    ).reset_index()
    
    
    return weekly_rentals



def get_average_hourly_rentals_df(data):
    
    hourly_data = data.copy()
    # Group by hour and sum rentals
    hourly_rentals = hourly_data.groupby('hour')['rented_bike_count'].mean().reset_index()


    return hourly_rentals



def rentals_per_day(data, day, month, year):
    
    hour_data = data.copy()
    
    hour_data = hour_data[(hour_data["day"] == day) & (hour_data["month"] == month) & (hour_data["year"] == year)]
    
    return hour_data


def get_avg_temp_humidity():
    
    bike_data = get_bikeshare_data()
    bike_data["date"] = pd.to_datetime(bike_data["date"], dayfirst=True)
    
    temp_humidity = bike_data.copy()
    
    temp_humidity["year_month"] = temp_humidity["date"].dt.strftime('%Y-%m')
    
    averages = temp_humidity.groupby('year_month').agg(avg_temperature = ('temp', 'mean'), avg_humidity = ('humidity', 'mean')).reset_index()
    
    # set index to year_month
    averages.set_index('year_month', inplace=True)
    
    return averages