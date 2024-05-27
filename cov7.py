# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:43:13 2024

@author: HP-PC
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_excel("D:/PHDS/Data Science Fundamentals/Assignments/Visualization/COVID_worldwide.xlsx")

# Set the title of the Streamlit app
st.title("Global COVID-19 Data Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# Filter data based on user selection
selected_continent = st.sidebar.multiselect('Select Continent', df['continentExp'].unique())
if selected_continent:
    filtered_countries = df[df['continentExp'].isin(selected_continent)]['countriesAndTerritories'].unique()
    selected_country = st.sidebar.multiselect('Select Country', filtered_countries)
else:
    selected_country = st.sidebar.multiselect('Select Country', df['countriesAndTerritories'].unique())

selected_year = st.sidebar.multiselect('Select Year', df['year'].unique())
if selected_year:
    filtered_months = df[df['year'].isin(selected_year)]['dateRep'].dt.strftime('%b-%Y').unique()
    selected_month = st.sidebar.multiselect('Select Month', filtered_months)
else:
    selected_month = st.sidebar.multiselect('Select Month', df['dateRep'].dt.strftime('%b-%Y').unique())

# Apply filters
if selected_continent:
    df = df[df['continentExp'].isin(selected_continent)]
if selected_country:
    df = df[df['countriesAndTerritories'].isin(selected_country)]
if selected_year:
    df = df[df['year'].isin(selected_year)]
if selected_month:
    df = df[df['dateRep'].dt.strftime('%b-%Y').isin(selected_month)]

# Display the data frame
st.subheader("COVID-19 Data")
st.write(df.head())

# Calculate mean and sum values for the first table
mean_cases = df['cases'].mean()
mean_deaths = df['deaths'].mean()
sum_cases = df['cases'].sum()
sum_deaths = df['deaths'].sum()

# Helper function to format large numbers consistently
def human_format(num, pos=None):
    if num >= 1e6:
        return f'{num*1e-6:.1f}M'
    elif num >= 1e3:
        return f'{num*1e-3:.1f}K'
    else:
        return str(int(num))

# Display mean and sum data boxes only for the first table
col1, col2 = st.columns(2)

with col1:
    st.info(f"Mean Cases: {mean_cases:.2f}")
    st.info(f"Total Cases: {human_format(sum_cases)}")

with col2:
    st.info(f"Mean Deaths: {mean_deaths:.2f}")
    st.info(f"Total Deaths: {human_format(sum_deaths)}")



# Visualization 1: COVID-19 Cases by Country and Deaths by Country (Side by Side)
st.subheader("COVID-19 Cases and Deaths by Country")

# Calculate top countries cases and deaths
top_countries_cases = df.groupby('countriesAndTerritories')['cases'].sum().nlargest(10)
top_countries_deaths = df.groupby('countriesAndTerritories')['deaths'].sum().nlargest(10)

# Plot top countries cases and deaths
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
top_countries_cases.plot(kind='bar', ax=ax1, color='skyblue')
top_countries_deaths.plot(kind='bar', ax=ax2, color='salmon')

# Set titles and labels
ax1.set_title("Top 10 Countries by COVID-19 Cases")
ax1.set_ylabel("Total Cases (in 100,000)")
ax1.set_xlabel("Countries")
ax1.set_yticklabels([f'{int(y/1e5):,}' for y in ax1.get_yticks()])
ax2.set_title("Top 10 Countries by COVID-19 Deaths")
ax2.set_ylabel("Total Deaths")
ax2.set_xlabel("Countries")
fig.suptitle("Figure 1: Top 10 Countries by COVID-19 Cases and Deaths", fontsize=16)

# Display the visualization
st.pyplot(fig)

# Visualization 3: Weekly Statistics and Visualization 4: Rate of Increase (Side by Side)
st.subheader("Weekly COVID-19 Cases and Rate of Increase")

df['week'] = df['dateRep'].dt.isocalendar().week
weekly_cases = df.groupby('week')['cases'].sum()
df_sorted = df.sort_values(by='dateRep')
df_sorted['cumulative_cases'] = df_sorted.groupby('countriesAndTerritories')['cases'].cumsum()
df_sorted['rate_of_increase'] = df_sorted['cumulative_cases'].pct_change().fillna(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

weekly_cases.plot(ax=ax1, label='Cases', color='blue')
ax1.set_title("Weekly COVID-19 Cases")
ax1.set_ylabel("Total Cases (in 100,000)")
ax1.set_xlabel("Week Number")
ax1.set_yticklabels([f'{int(y/1e5):,}' for y in ax1.get_yticks()])

df_sorted.groupby('dateRep')['rate_of_increase'].mean().plot(ax=ax2, color='red')
ax2.set_title("Average Rate of Increase of COVID-19 Cases")
ax2.set_ylabel("Rate of Increase")
ax2.set_xlabel("Date")

fig.suptitle("Figure 3: Weekly COVID-19 Cases and Rate of Increase", fontsize=16)
st.pyplot(fig)

# Visualization 5: Cases and Deaths per 100,000 Population and Scatterplot
st.subheader("COVID-19 Cases and Deaths per 100,000 Population")
df['cases_per_100k'] = df['cases'] * 100000 / df['popData2019']
df['deaths_per_100k'] = df['deaths'] * 100000 / df['popData2019']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

df.groupby('dateRep')['cases_per_100k'].mean().plot(ax=ax1, label='Cases', color='purple')
df.groupby('dateRep')['deaths_per_100k'].mean().plot(ax=ax1, label='Deaths', color='orange')
ax1.set_title("COVID-19 Cases and Deaths per 100,000 Population")
ax1.set_ylabel("Rate per 100,000")
ax1.set_xlabel("Date")
ax1.legend()

sns.scatterplot(x='cases_per_100k', y='deaths_per_100k', data=df, ax=ax2)
sns.regplot(x='cases_per_100k', y='deaths_per_100k', data=df, scatter=False, ax=ax2, color='red', label='Trend line')
ax2.set_title("Scatterplot of Cases vs Deaths per 100,000 Population")
ax2.set_xlabel("Cases per 100,000")
ax2.set_ylabel("Deaths per 100,000")
ax2.legend()

fig.suptitle("Figure 4: COVID-19 Cases and Deaths per 100,000 Population and Scatterplot", fontsize=16)
st.pyplot(fig)

# Visualization 6: COVID-19 Cases and Deaths per 100,000 Population by Continent and Monthly CFR Trends (Side by Side)
st.subheader("COVID-19 Cases and Deaths per 100,000 Population by Continent and Trends of COVID-19 Case Fatality Rate over Months")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

df_continent = df.groupby('continentExp')[['cases_per_100k', 'deaths_per_100k']].sum().sort_values(by='cases_per_100k', ascending=False)
df_continent.plot(kind='bar', ax=ax1)
ax1.set_title("COVID-19 Cases and Deaths per 100,000 Population by Continent")
ax1.set_ylabel("Rate per 100,000")
ax1.set_xlabel("Continents")
ax1.legend(['Cases', 'Deaths'])
for p in ax1.patches:
    ax1.annotate(f'{human_format(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

df['month'] = df['dateRep'].dt.to_period('M')
monthly_cfr = df.groupby('month')['deaths'].sum() / df.groupby('month')['cases'].sum() * 100

monthly_cfr.plot(ax=ax2, marker='o', color='red')
ax2.set_title("Trends of COVID-19 Case Fatality Rate over Months")
ax2.set_ylabel("Case Fatality Rate (%)")
ax2.set_xlabel("Month")
ax2.set_xticklabels([label.strftime('%b %Y') for label in monthly_cfr.index.to_timestamp()], rotation=45)
for idx, val in monthly_cfr.items():
    ax2.text(idx.to_timestamp(), val, f'{val:.2f}', ha='center', va='bottom')

fig.suptitle("Figure 5: COVID-19 Cases and Deaths per 100,000 Population by Continent and Trends of COVID-19 Case Fatality Rate over Months", fontsize=16)
st.pyplot(fig)
