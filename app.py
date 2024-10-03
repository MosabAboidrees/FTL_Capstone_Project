import streamlit as st
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import holidays
from datetime import datetime, timedelta
from tabulate import tabulate
from scipy.stats import chi2_contingency
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.feature_selection import VarianceThreshold

# Set the page layout
st.set_page_config(layout="wide")

# Title
st.title("Energy Consumption Prediction")

# Background Information
st.markdown("""
## Project Background

This project focuses on predicting hourly energy consumption across various electric companies in the United States. Accurate energy consumption forecasts are vital for ensuring reliable power grid management and efficient resource allocation. By using historical energy consumption data, we aim to develop a machine learning model that can predict future energy demands.

**Key Objectives:**
- Develop a predictive model for energy consumption.
- Understand how factors like time of day, season, and holidays influence energy consumption.
- Provide actionable insights to assist in grid management and resource planning.
""")


# Sidebar for uploading zip file
st.sidebar.header("Upload Data")

# Uploading the zip file
uploaded_file = st.sidebar.file_uploader("Choose a ZIP file", type="zip")

if uploaded_file is not None:
    # Extract the zip file
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall('/tmp/energy_data')  # Extract to a temporary directory
    
    # Load the datasets
    file_names = [
        'AEP_hourly.csv', 'COMED_hourly.csv', 'DAYTON_hourly.csv',
        'DEOK_hourly.csv', 'DOM_hourly.csv', 'DUQ_hourly.csv',
        'EKPC_hourly.csv', 'FE_hourly.csv', 'NI_hourly.csv',
        'PJM_Load_hourly.csv', 'PJME_hourly.csv', 'PJMW_hourly.csv'
    ]
    
    df_raw = {}
    for file_name in file_names:
        key = file_name.split('_')[0].lower()
        df_raw[key] = pd.read_csv(f'/tmp/energy_data/{file_name}', low_memory=False)
    
    # Start of Data Processing Section
    st.markdown("---")
    st.title('Energy Consumption Data Processing')

    # Description before the data processing
    st.write("""
    ## Data Processing Steps

    In this section, we combine energy consumption data from various electric companies into a single DataFrame. 
    This involves adding a new column indicating the electric company for each dataset and standardizing the column names for energy consumption.

    Here are the steps involved:

    1. **Add Electric Company Column**: Each dataset gets a new column labeled `electric_company`, identifying the data source.
    2. **Rename Energy Consumption Columns**: The energy consumption column is renamed consistently across all datasets to `mw_energy_consumption`.
    3. **Combine DataFrames**: All the individual DataFrames are concatenated into a single DataFrame, `combined_df`.

    Let's explore the resulting combined data:
    """)

    # Line before processing
    st.markdown("---")

    df1 = df_raw.copy()

    # Step 1: Add the electric company column
    for key in df1.keys():
        df1[key]['electric_company'] = key.upper()

    # Step 2: Rename the energy consumption columns
    rename_dict = {
        'aep': 'AEP_MW', 'comed': 'COMED_MW', 'dayton': 'DAYTON_MW',
        'deok': 'DEOK_MW', 'dom': 'DOM_MW', 'duq': 'DUQ_MW',
        'ekpc': 'EKPC_MW', 'fe': 'FE_MW', 'ni': 'NI_MW',
        'pjm': 'PJM_Load_MW', 'pjme': 'PJME_MW', 'pjmw': 'PJMW_MW'
    }

    for key, original_column in rename_dict.items():
        df1[key] = df1[key].rename(columns={original_column: 'mw_energy_consumption'})

    # Step 3: Combine all the dataframes into one
    combined_df = pd.concat(df1.values(), ignore_index=True)

    # Streamlit display
    st.title('Combined Energy Consumption Data')

    # Description before displaying data
    st.write("""
    The combined DataFrame includes energy consumption data from multiple companies. 
    Below, you can see the first few rows of this unified dataset, followed by a random sample of entries to illustrate data diversity.
    """)

    # Display the combined dataframe shape
    st.write("### Data Shape: ", combined_df.shape)

    # Display the combined dataframe head
    st.subheader("Head of Combined DataFrame")
    st.dataframe(combined_df.head())

    # Display random samples from the combined dataframe
    st.subheader("Random Sample from Combined DataFrame")
    st.dataframe(combined_df.sample(5))

    # Line after displaying data
    st.markdown("---")

    # End of Data Processing Section
    st.write("""
    This data preparation step ensures that the energy consumption data is ready for further analysis and modeling. 
    The consistency in column names and the inclusion of the electric company identifier will facilitate better insights into energy usage patterns across different regions.
    """)
    st.markdown("---")

    # Start of Part 1: DESCRIPTION OF DATA
    
    st.title("Part 1: DESCRIPTION OF DATA")

    # Description of Data
    st.write("""
    In this section, we perform an initial exploration of the combined energy consumption data. 
    The aim is to understand the structure and key characteristics of the dataset. 
    We will rename columns, check data dimensions, data types, missing values, and perform descriptive statistics.
    """)

    # Make a copy of combined_df
    df1 = combined_df.copy()

    # Show current columns
    st.subheader("Current Columns")
    st.write("These are the current columns in the DataFrame:")
    st.write(df1.columns.tolist())

    # Step 1.1: Rename Columns
    st.subheader("1.1 Rename Columns")
    st.write("""
    We rename the columns to a consistent format to ensure clarity and ease of use. 
    The changes are as follows:
    - **Datetime** -> **datetime**
    - **mw_energy_consumption** -> **mw_energy_consumption** (remains the same)
    - **electric_company** -> **electric_company** (remains the same)
    """)

    # Rename columns
    cols_old = ['Datetime', 'mw_energy_consumption', 'electric_company']
    cols_new = ['datetime', 'mw_energy_consumption', 'electric_company']
    df1.columns = cols_new

    # Display the new column names
    st.write("New column names:")
    st.write(df1.columns.tolist())

    # Step 1.2: Data Dimensions
    st.subheader("1.2 Data Dimensions")
    st.write("""
    Understanding the dimensions of the dataset gives us insight into the size and complexity of the data we are working with.
    """)
    # Check data dimensions
    st.write(f"Number of Rows: {df1.shape[0]}")
    st.write(f"Number of Columns: {df1.shape[1]}")

    # Step 1.3: Data Types
    st.subheader("1.3 Data Types")
    st.write("""
    Checking the data types is crucial for ensuring that operations on the data are performed correctly. 
    We ensure that the datetime column is of datetime type.
    """)
    # Display data types
    st.write("Data types before conversion:")
    st.write(df1.dtypes)

    # Convert 'datetime' column to datetime type
    df1['datetime'] = pd.to_datetime(df1['datetime'])

    # Display data types after conversion
    st.write("Data types after conversion:")
    st.write(df1.dtypes)

    # Step 1.4: Check NA
    st.subheader("1.4 Check Missing Values")
    st.write("""
    Missing values can lead to inaccurate analysis and must be identified early on. 
    Here we count the number of missing values in each column.
    """)
    # Check for missing values
    missing_values = df1.isna().sum()
    st.write("Missing values in each column:")
    st.write(missing_values)

    # Step 1.5: Descriptive Statistics
    st.subheader("1.5 Descriptive Statistics")

    # Divide into numerical and categorical columns
    num_attributes = df1.select_dtypes(include=['float64'])
    cat_attributes = df1.select_dtypes(exclude=['float64', 'datetime64[ns]'])

    # Sample numerical attributes
    st.write("Sample of numerical attributes:")
    st.dataframe(num_attributes.sample(5))

    # Sample categorical attributes
    st.write("Sample of categorical attributes:")
    st.dataframe(cat_attributes.sample(5))

    # Step 1.6: Numerical Attributes
    st.subheader("1.6 Numerical Attributes")
    st.write("""
    Descriptive statistics provide insights into the distribution and variation within numerical attributes.
    We calculate measures of central tendency and dispersion for numerical attributes.
    """)

    # Central Tendency - mean, median
    ct1 = pd.DataFrame(num_attributes.apply(np.mean)).T
    ct2 = pd.DataFrame(num_attributes.apply(np.median)).T

    # Dispersion - std, min, max, range, skew, kurtosis
    d1 = pd.DataFrame(num_attributes.apply(np.std)).T
    d2 = pd.DataFrame(num_attributes.apply(min)).T
    d3 = pd.DataFrame(num_attributes.apply(max)).T
    d4 = pd.DataFrame(num_attributes.apply(lambda x: x.max() - x.min())).T
    d5 = pd.DataFrame(num_attributes.apply(lambda x: x.skew())).T
    d6 = pd.DataFrame(num_attributes.apply(lambda x: x.kurtosis())).T

    # Concatenate
    m = pd.concat([d2, d3, d4, ct1, ct2, d1, d5, d6]).T.reset_index()
    m.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']

    # Display the descriptive statistics
    st.write("Descriptive statistics for numerical attributes:")
    st.dataframe(m)

    # Plot distribution of mw_energy_consumption
    st.write("Distribution of Megawatt Energy Consumption:")
    fig, ax = plt.subplots()
    sns.histplot(df1['mw_energy_consumption'], kde=True, ax=ax)
    plt.title('Distribution of Megawatt Energy Consumption')
    st.pyplot(fig)

    # Step 1.7: Categorical Attributes
    st.subheader("1.7 Categorical Attributes")
    st.write("""
    Descriptive statistics for categorical attributes can help us understand the diversity and frequency of categorical data points.
    """)

    # Descriptive statistics for categorical attributes
    unique_counts = cat_attributes.apply(lambda x: x.unique().shape[0])
    st.write("Unique value counts for categorical attributes:")
    st.write(unique_counts)

    # Plot boxplot for mw_energy_consumption by electric_company
    st.write("Megawatt Energy Consumption by Electric Company:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='electric_company', y='mw_energy_consumption', data=combined_df, ax=ax)
    plt.title('Megawatt Energy Consumption by Electric Company')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # End of Part 1: DESCRIPTION OF DATA
    st.markdown("---")
    st.write("""
    The exploration of the dataset has provided us with a foundational understanding of its structure and key characteristics. 
    These insights will guide further analysis and modeling steps.
    """)
    st.markdown("---")

    # Start of Part 2: FEATURE ENGINEERING

    st.title("Part 2: FEATURE ENGINEERING")

    st.write("""
    In this section, we enhance the dataset by deriving new features from existing columns.
    Feature engineering helps us uncover hidden patterns and relationships in the data, improving the model's predictive power.
    """)

    # Step 2.1: Copy the dataset
    st.subheader("2.1 Copy Dataset")
    st.write("""
    We begin by making a copy of the existing dataset to ensure the original data remains unchanged.
    """)
    df2 = df1.copy()
    st.write("Sample data from the copied dataset:")
    st.dataframe(df2.sample(5))

    # Step 2.2: Extract Date Components
    st.subheader("2.2 Extract Date Components")
    st.write("""
    We extract date-related features such as date, year, month, and hour from the datetime column.
    These components will allow us to analyze temporal patterns in energy consumption.
    """)

    # Extract date, year, month, hour
    df2['date'] = df2['datetime'].dt.date
    df2['year'] = df2['datetime'].dt.year
    df2['month'] = df2['datetime'].dt.month
    df2['hour_of_day'] = df2['datetime'].dt.hour

    # Display extracted date components
    st.write("Extracted Date Components:")
    st.write(df2[['datetime', 'date', 'year', 'month', 'hour_of_day']].head())

    # Step 2.3: Determine Seasons
    st.subheader("2.3 Determine Seasons")
    st.write("""
    Based on the month, we assign a season to each record. 
    Seasons are defined as follows:
    - **Winter**: December, January, February
    - **Spring**: March, April, May
    - **Summer**: June, July, August
    - **Autumn**: September, October, November
    """)

    # Determine the season
    df2['season'] = df2['datetime'].apply(lambda x: 'Winter' if x.month in [12, 1, 2]
                                          else 'Spring' if x.month in [3, 4, 5]
                                          else 'Summer' if x.month in [6, 7, 8]
                                          else 'Autumn')

    # Display season assignments
    st.write("Season Assignment:")
    st.write(df2[['datetime', 'month', 'season']].head())

    # Step 2.4: Identify Holidays
    st.subheader("2.4 Identify Holidays")
    st.write("""
    Using the `holidays` library, we determine if each date falls on a US holiday or the day before a holiday.
    This feature helps capture patterns in energy consumption related to holidays.
    """)

    # Determine holidays
    us_holidays = holidays.US()
    df2['holidays'] = df2['datetime'].apply(lambda x: 'Holiday' if x.date() in us_holidays
                                            else 'Holiday' if (x + pd.DateOffset(days=1)).date() in us_holidays
                                            else 'Normal day')

    # Display holiday assignments
    st.write("Holiday Assignment:")
    st.write(df2[['datetime', 'holidays']].head())

    # Step 2.5: Extract Day of the Week
    st.subheader("2.5 Extract Day of the Week")
    st.write("""
    We extract the day of the week from the datetime column, represented as an integer (0 for Monday, 6 for Sunday).
    This feature will help analyze weekly consumption patterns.
    """)

    # Extract day of the week
    df2['day_of_week'] = df2['datetime'].dt.weekday

    # Display day of the week extraction
    st.write("Day of the Week:")
    st.write(df2[['datetime', 'day_of_week']].head())

    # Step 2.6: Recheck Data Types
    st.subheader("2.6 Recheck Data Types")
    st.write("""
    After adding new features, we recheck the data types to ensure correctness, especially for date and time-related features.
    """)

    # Recheck data types after feature engineering
    st.write("Data types before conversion:")
    st.write(df2.dtypes)

    # Convert 'date' to datetime
    df2['date'] = pd.to_datetime(df2['date'])

    # Recheck data types
    st.write("Data types after conversion:")
    st.write(df2.dtypes)

    # Step 2.7: Separate Numerical and Categorical Attributes
    st.subheader("2.7 Separate Numerical and Categorical Attributes")
    st.write("""
    Separating numerical and categorical attributes helps us prepare for exploratory data analysis and modeling.
    """)

    # Separate numerical and categorical attributes
    num_attributes = df2.select_dtypes(include=['int32', 'int64', 'float64'])
    cat_attributes = df2.select_dtypes(exclude=['int32', 'int64', 'float64', 'datetime64[ns]'])

    # Display samples
    st.write("Sample of Numerical Attributes:")
    st.dataframe(num_attributes.sample(5))

    st.write("Sample of Categorical Attributes:")
    st.dataframe(cat_attributes.sample(5))

    # Step 2.8: Variable Filtering
    st.subheader("2.8 Variable Filtering")
    st.write("""
    We make a copy of the dataset for variable filtering. This step allows us to apply specific filters or transformations for analysis and modeling.
    """)

    # Make a copy for variable filtering
    df3 = df2.copy()
    st.write("Sample data from the filtered dataset:")
    st.dataframe(df3.sample(5))

    # End of Part 2: FEATURE ENGINEERING
    st.markdown("---")
    st.write("""
    Feature engineering is a critical step in preparing data for machine learning models. 
    By extracting and transforming relevant features, we can enhance the model's ability to learn patterns from the data.
    """)
    st.markdown("---")

    # Helper function for Cramér's V
    def cramers_v(x, y):
        """Calculate Cramér's V statistic for categorical-categorical association."""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1)**2) / (n - 1)
        kcorr = k - ((k - 1)**2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    # Start of Part 3: EXPLORATORY DATA ANALYSIS

    st.title("Part 3: EXPLORATORY DATA ANALYSIS AFTER FEATURE ENGINEERING")

    st.write("""
    This section performs exploratory data analysis (EDA) to uncover patterns, trends, and insights within the dataset. 
    EDA helps understand the data better and guides feature selection for modeling.
    """)

    # Step 3.1: Univariate Analysis
    st.subheader("3.1 Univariate Analysis")
    st.write("""
    Univariate analysis examines individual variables to understand their distributions and key characteristics.
    """)

    # Make a copy for analysis
    df4 = df3.copy()

    # 3.1.1 Response Variable
    st.subheader("3.1.1 Response Variable")
    st.write("""
    The distribution of the response variable, `mw_energy_consumption`, is analyzed using a histogram.
    This helps us understand the frequency and spread of energy consumption values.
    """)
    fig, ax = plt.subplots()
    sns.histplot(df4['mw_energy_consumption'], kde=False, ax=ax)
    ax.set_title('Distribution of Megawatt Energy Consumption')
    st.pyplot(fig)

    # 3.1.2 Numerical Variables
    st.subheader("3.1.2 Numerical Variables")
    st.write("""
    Histograms for numerical variables reveal their distributions and potential outliers.
    """)
    # Numerical Variable
    fig, axs = plt.subplots(figsize=(10, 5))
    df4[['mw_energy_consumption','year', 'month', 'day_of_week','hour_of_day']].hist(bins=25, ax=axs)
    plt.tight_layout()
    st.pyplot(fig)

    # 3.1.3 Categorical Variables
    st.subheader("3.1.3 Categorical Variables")
    st.write("""
    A snapshot of categorical variables gives insight into unique categories and their frequency distributions.
    """)
    cat_attributes = df4.select_dtypes(exclude=['float64', 'datetime64[ns]'])
    st.write("Sample of Categorical Attributes:")
    st.dataframe(cat_attributes.sample(5))

    st.write("""
    Using `drop_duplicates()` on the `season` and `holidays` columns identifies unique categories, aiding EDA and feature engineering.
    """)

    # Check unique values
    unique_electric_company = df4['electric_company'].drop_duplicates()
    unique_season = df4['season'].drop_duplicates()
    unique_holidays = df4['holidays'].drop_duplicates()

    st.write("Unique Electric Companies:", unique_electric_company.tolist())
    st.write("Unique Seasons:", unique_season.tolist())
    st.write("Unique Holidays:", unique_holidays.tolist())

    # Countplot and KDE plots for categorical variables
    st.subheader("Categorical Variable Distributions")
    st.write("""
    Countplots and KDE plots provide insights into the distribution and density of categorical variables.
    """)
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))

    # Electric Company
    sns.countplot(df4['electric_company'], ax=axs[0, 0])
    axs[0, 0].set_title('Count of Records by Electric Company')

    for company in df4['electric_company'].unique():
        sns.kdeplot(df4[df4['electric_company'] == company]['mw_energy_consumption'], label=company, shade=True, ax=axs[0, 1])
    axs[0, 1].set_title('Energy Consumption Distribution by Electric Company')
    axs[0, 1].legend()

    # Season
    sns.countplot(df4['season'], ax=axs[1, 0])
    axs[1, 0].set_title('Count of Records by Season')

    for season in df4['season'].unique():
        sns.kdeplot(df4[df4['season'] == season]['mw_energy_consumption'], label=season, shade=True, ax=axs[1, 1])
    axs[1, 1].set_title('Energy Consumption Distribution by Season')
    axs[1, 1].legend()

    # Holidays
    sns.countplot(df4['holidays'], ax=axs[2, 0])
    axs[2, 0].set_title('Count of Records by Holidays')

    for holiday in df4['holidays'].unique():
        sns.kdeplot(df4[df4['holidays'] == holiday]['mw_energy_consumption'], label=holiday, shade=True, ax=axs[2, 1])
    axs[2, 1].set_title('Energy Consumption Distribution by Holidays')
    axs[2, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Step 3.2: Bivariate Analysis
    st.subheader("3.2 Bivariate Analysis")
    st.write("""
    Bivariate analysis investigates relationships between pairs of variables, uncovering dependencies and interactions.
    """)

    # Hypotheses
    st.write("""
    **1. Creation of Hypotheses**

    1.1 Demographic Hypotheses

    - Older people spend less energy
    - Larger families spend more energy
    - Female people spend more energy
    - Family activity influences energy expenditure in the early morning and late afternoon.

    1.2 Geographic Hypotheses

    - Seasons with higher temperatures use more energy
    - Locations present in geographic accidents spend more energy
    - Climate with higher temperatures use more energy

    1.3 Sociocultural Hypotheses

    - Routines that start when there is less sunlight use more energy
    - Tribes spend less energy
    - Nations spend more energy
    - Communities spend less energy
    - Holiday periods spend more energy

    1.4 Final List of Hypotheses

    - Family activity influences energy expenditure in the early morning and late afternoon only
    - Seasons with higher or lower temperatures use more energy
    - Holiday periods spend more energy
    - Weekends periods spend more energy
    """)

    # H1: Family activity influences energy expenditure in the early morning and late afternoon only
    st.subheader("H1: Family Activity Influence on Energy Expenditure")
    st.write("""
    **Hypothesis:** Family activities influence energy consumption in the early morning and late afternoon.

    **Analysis:** The plot shows energy consumption trends throughout the day, revealing key consumption patterns.

    **Relevance:** High - Understanding daily patterns helps manage peak loads and grid stability.

    **Conclusion:** False - Energy consumption is significant throughout the day, not just in the morning and afternoon.
    """)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    aux1 = df4[['hour_of_day', 'mw_energy_consumption']].groupby('hour_of_day').sum().reset_index()
    sns.barplot(x='hour_of_day', y='mw_energy_consumption', data=aux1, ax=axs[0])
    axs[0].set_title('Total Energy Consumption by Hour of Day')

    sns.regplot(x='hour_of_day', y='mw_energy_consumption', data=aux1, ax=axs[1])
    axs[1].set_title('Regression Plot of Energy Consumption by Hour of Day')

    sns.heatmap(aux1.corr(method='pearson'), annot=True, ax=axs[2])
    axs[2].set_title('Correlation Heatmap')

    plt.tight_layout()
    st.pyplot(fig)

    # H2: Seasons with higher or lower temperatures use more energy
    st.subheader("H2: Energy Use by Season")
    st.write("""
    **Hypothesis:** Seasons with extreme temperatures (winter and summer) result in higher energy consumption.

    **Analysis:** The plots demonstrate that winter and summer have higher energy consumption than other seasons.

    **Conclusion:** True - Seasons with extreme temperatures do have higher energy consumption.

    **Relevance:** Medium - Understanding seasonal variations aids utilities in planning for seasonal demand.
    """)

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    aux1 = df4[['season', 'mw_energy_consumption']].groupby('season').sum().reset_index()
    sns.barplot(x='season', y='mw_energy_consumption', data=aux1, order=['Winter', 'Spring', 'Summer', 'Autumn'], ax=axs[0])
    axs[0].set_title('Total Energy Consumption by Season')

    aux2 = df4[['year', 'season', 'mw_energy_consumption']].groupby(['year', 'season']).sum().reset_index()
    sns.barplot(x='year', y='mw_energy_consumption', hue='season', data=aux2, hue_order=['Winter', 'Spring', 'Summer', 'Autumn'], ax=axs[1])
    axs[1].set_title('Yearly Total Energy Consumption by Season')

    plt.tight_layout()
    st.pyplot(fig)

    # H3: Holiday periods spend more energy
    st.subheader("H3: Holiday Energy Consumption")
    st.write("""
    **Hypothesis:** Energy consumption is higher during holiday periods.

    **Analysis:** The analysis shows that energy consumption is lower during holidays compared to normal days.

    **Conclusion:** False - Holiday periods spend less energy.

    **Relevance:** Medium - Understanding holiday patterns aids utilities in planning for reduced demand.
    """)

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    aux1 = df4[['holidays', 'mw_energy_consumption']].groupby('holidays').sum().reset_index()
    sns.barplot(x='holidays', y='mw_energy_consumption', data=aux1, ax=axs[0])
    axs[0].set_title('Total Energy Consumption by Holidays')

    aux2 = df4[['year', 'holidays', 'mw_energy_consumption']].groupby(['year', 'holidays']).sum().reset_index()
    sns.barplot(x='year', y='mw_energy_consumption', hue='holidays', data=aux2, ax=axs[1])
    axs[1].set_title('Yearly Total Energy Consumption by Holidays')

    plt.tight_layout()
    st.pyplot(fig)

    # H4: Weekends periods spend more energy
    st.subheader("H4: Weekend Energy Consumption")
    st.write("""
    **Hypothesis:** Energy consumption is higher during weekends.

    **Analysis:** The analysis indicates that energy consumption is lower on weekends compared to weekdays.

    **Conclusion:** False - Weekends periods spend less energy.

    **Relevance:** High - Understanding weekly patterns helps manage daily fluctuations and grid stability.
    """)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    aux1 = df4[['day_of_week', 'mw_energy_consumption']].groupby('day_of_week').sum().reset_index()
    sns.barplot(x='day_of_week', y='mw_energy_consumption', data=aux1, ax=axs[0, 0])
    axs[0, 0].set_title('Total Energy Consumption by Day of Week')

    aux2 = df4[['month', 'day_of_week', 'mw_energy_consumption']].groupby(['month', 'day_of_week']).sum().reset_index()
    sns.barplot(x='month', y='mw_energy_consumption', hue='day_of_week', data=aux2, ax=axs[0, 1])
    axs[0, 1].set_title('Monthly Total Energy Consumption by Day of Week')

    sns.regplot(x='day_of_week', y='mw_energy_consumption', data=aux1, ax=axs[1, 0])
    axs[1, 0].set_title('Regression Plot of Energy Consumption by Day of Week')

    sns.heatmap(aux1.corr(method='pearson'), annot=True, ax=axs[1, 1])
    axs[1, 1].set_title('Correlation Heatmap')

    # Add legend for day of week
    day_of_week_legend = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    st.write("Day of the Week Legend:", day_of_week_legend)

    plt.tight_layout()
    st.pyplot(fig)

    # Summary of Hypotheses
    st.subheader("Summary of Hypotheses")
    st.write("""
    The table below summarizes the hypotheses, conclusions, and their relevance:
    """)
    tab = [['Hypotheses', 'Conclusion', 'Relevance'],
          ['H1', 'False', 'High'],
          ['H2', 'True', 'Medium'],
          ['H3', 'False', 'Medium'],
          ['H4', 'False', 'High']
          ]
    st.write(tabulate(tab, headers='firstrow', tablefmt='grid'))

    # Multivariate Analysis
    st.subheader("3.3 Multivariate Analysis")
    st.write("""
    Multivariate analysis explores relationships between multiple variables simultaneously, providing a holistic view of interactions.
    """)

    # 3.3.1 Numerical Attributes
    st.subheader("3.3.1 Numerical Attributes")
    st.write("""
    The correlation matrix for numerical attributes is calculated using Pearson's correlation coefficient, revealing relationships between numerical variables.
    """)
    num_attributes = df4.select_dtypes(include=['int32', 'int64', 'float64'])
    correlation = num_attributes.corr(method='pearson')

    fig, ax = plt.subplots()
    sns.heatmap(correlation, annot=True, ax=ax)
    ax.set_title('Correlation Heatmap for Numerical Attributes')
    st.pyplot(fig)

    # 3.3.2 Categorical Attributes
    st.subheader("3.3.2 Categorical Attributes")
    st.write("""
    Cramér's V is calculated for each pair of categorical variables to assess the strength of association between them.
    """)
    cat_data = ['electric_company', 'season', 'holidays']
    cramer_v_matrix = pd.DataFrame(index=cat_data, columns=cat_data)

    for col1 in cat_data:
        for col2 in cat_data:
            cramer_v_matrix.loc[col1, col2] = cramers_v(cat_attributes[col1], cat_attributes[col2])

    # Convert the results to float for plotting
    cramer_v_matrix = cramer_v_matrix.astype(float)

    fig, ax = plt.subplots()
    sns.heatmap(cramer_v_matrix, annot=True, ax=ax)
    ax.set_title("Cramér's V Matrix for Categorical Attributes")
    st.pyplot(fig)

    # End of Part 3: EXPLORATORY DATA ANALYSIS
    st.markdown("---")
    st.write("""
    Exploratory data analysis has provided valuable insights into the dataset, helping identify patterns, relationships, and potential areas for further exploration. These findings will guide the feature selection and model-building process.
    """)
    st.markdown("---")

    # Start of Part 4: Feature Selection
    st.markdown("---")
    st.title("Part 4: Feature Selection")

    st.write("""
    This section focuses on selecting the most relevant features for the model. 
    The goal is to identify the features that contribute most to predicting energy consumption and to prepare the data for modeling.
    """)

    # Step 4.1: Copy the Dataset
    st.subheader("4.1 Copy the Dataset")
    st.write("""
    We start by making a copy of the existing dataset to ensure that the original data remains unchanged.
    """)

    st.write("""
    The only categorical variable we will encode at this stage is "holidays". 
    Although the "season" variable is also categorical, it is cyclical in nature, and we will transform it later.
    Since the values in the "holidays" variable do not have any significant ordinal relationship and our 
    final dataset will not have too many columns, we will apply the "One Hot Encoding" technique. 
    This approach will allow us to convert the categorical data into a numerical format without implying 
    any rank or order among the categories.
    """)

    df5 = df4.copy()
    df5 = pd.get_dummies(df5, prefix=['holidays'], columns=['holidays'])

    df6 = df5.copy()
    st.write("Shape of the dataset:", df6.shape)

    # Step 4.2: Drop Unnecessary Columns
    st.subheader("4.2 Drop Unnecessary Columns")
    st.write("""
    We drop columns that were used to generate new features during feature engineering and are no longer needed.
    These columns include year, month, hour_of_day, season, and day_of_week.
    """)
    cols_drop = ['year', 'month', 'hour_of_day', 'season', 'day_of_week']
    df6 = df6.drop(cols_drop, axis=1)
    st.write("Columns after dropping unnecessary ones:")
    st.dataframe(df6.head())

    # Step 4.3: Determine Cutoff Dates for Training and Test Sets
    st.subheader("4.3 Determine Cutoff Dates for Training and Test Sets")
    st.write("""
    In this time-series project, we aim to forecast energy consumption for the next 6 months. 
    To achieve this, we separate the dataset into training and test sets, considering the datetime variable.
    The most recent 6 months will be in the test dataset, while the remaining data will be used for training.
    """)

    # Determine the minimum and maximum dates for each electric company
    min_max = []
    for company in df6['electric_company'].unique():
        min_max.append([company, df6[df6['electric_company'] == company]['date'].min(), df6[df6['electric_company'] == company]['date'].max()])

    mm = pd.DataFrame(columns=['electric_company', 'min_date', 'max_date'], data=min_max)
    st.write("Minimum and Maximum Dates for Each Electric Company:")
    st.dataframe(mm)

    # Step 4.4: Determine Cutoff Dates
    st.subheader("4.4 Determine Cutoff Dates")
    st.write("""
    Since the minimum and maximum dates differ for each electric company, we stipulate cutoff dates for test and training data based on the following rules:

    - min_date <= TRAIN DATA < cut_date (max_date - 12 months)
    - cut_date (max_date - 12 months) <= TEST DATA <= max_date
    """)

    # Determine the cutoff date at each electrical company.
    cut_date = []
    for i in range(0, 12):
        cut_date.append([df6['electric_company'].unique()[i], df6[['electric_company', 'date']].groupby('electric_company').max().reset_index()['date'][i] - timedelta(days=12 * 30)])

    cd = pd.DataFrame(columns=['electric_company', 'cut_date'], data=cut_date)
    st.write("Cutoff Dates for Each Electric Company:")
    st.dataframe(cd)

    # Step 4.5: Split the Dataset
    st.subheader("4.5 Split the Dataset")
    st.write("""
    The dataset is split into training and test sets based on the cutoff dates, ensuring a clear separation of data for forecasting purposes.
    """)

    train_dfs = []
    test_dfs = []
    for company in df6['electric_company'].unique():
        cut = cd[cd['electric_company'] == company]['cut_date'].values[0]
        train_dfs.append(df6[(df6['electric_company'] == company) & (df6['date'] < cut)])
        test_dfs.append(df6[(df6['electric_company'] == company) & (df6['date'] >= cut)])

    X_train = pd.concat(train_dfs)
    X_test = pd.concat(test_dfs)

    y_train = X_train['mw_energy_consumption']
    y_test = X_test['mw_energy_consumption']

    st.write("Training Set Shape:", X_train.shape)
    st.write("Test Set Shape:", X_test.shape)

    # Step 4.6: Map Electric Company to Numerical Values
    st.subheader("4.6 Map Electric Company to Numerical Values")
    st.write("""
    To enable the algorithm to understand the electric company variable, we transform it from a string to a numerical representation.
    """)
    company_dict = {'AEP': 1, 'COMED': 2, 'DAYTON': 3, 'DEOK': 4, 'DOM': 5, 'DUQ': 6, 'EKPC': 7, 'FE': 8, 'NI': 9, 'PJM': 10, 'PJME': 11, 'PJMW': 12}
    X_train['electric_company'] = X_train['electric_company'].map(company_dict)
    X_test['electric_company'] = X_test['electric_company'].map(company_dict)

    st.write("Training Set with Numerical Electric Company:")
    st.dataframe(X_train.head())

    st.write("Test Set with Numerical Electric Company:")
    st.dataframe(X_test.head())

    # Step 4.7: Prepare Training and Test Datasets
    st.subheader("4.7 Prepare Training and Test Datasets")
    st.write("""
    The final step is to prepare the datasets by removing unnecessary columns and transforming them into the desired format.
    """)
    X_train_n = X_train.drop(['date', 'datetime', 'mw_energy_consumption'], axis=1).values
    y_train_n = y_train.values.ravel()

    st.write("Training Features Shape:", X_train_n.shape)
    st.write("Training Target Shape:", y_train_n.shape)

    # Feature Selection using Filter Method
    st.subheader("Feature Selection using Filter Method")
    st.write("""
    This method ranks features based on statistical measures, selecting the top-ranked ones.
    We use the Variance Threshold method to remove features with low variance (i.e., features that don't change much across samples).
    """)

    # Variance Threshold
    selector = VarianceThreshold(threshold=0.4)  # Adjust threshold as needed
    X_train_selected = selector.fit_transform(X_train_n)

    # Get the indices of the selected features
    selected_feature_indices = selector.get_support(indices=True)

    st.write("Selected Feature Indices:", selected_feature_indices)
    st.write("Selected Features Shape:", X_train_selected.shape)

    # Identify Best Features
    st.subheader("Identify Best Features")
    st.write("""
    The selected features are identified and compared with the features selected during exploratory data analysis.
    """)

    # Identify best features
    X_train_fs = X_train.drop(['date', 'datetime', 'mw_energy_consumption'], axis=1)
    cols_selected = X_train_fs.iloc[:, selected_feature_indices].columns.tolist()
    st.write("Selected Features:", cols_selected)

    # Identify not selected features
    cols_not_selected = list(np.setdiff1d(X_train_fs.columns, cols_selected))
    st.write("Not Selected Features:", cols_not_selected)

    # Confirming Selected Features
    st.subheader("Confirming Selected Features")
    st.write("""
    We confirm the selected features based on variance and compare them with the exploratory data analysis findings.
    """)

    # Table of Hypotheses
    # Data for the table
    tab = [
        ['Hypotheses', 'Conclusion', 'Relevance'],
        ['H1', 'False', 'High'],
        ['H2', 'True', 'Medium'],
        ['H3', 'False', 'Medium'],
        ['H4', 'False', 'High']
    ]

    # Create a DataFrame, excluding the header from tab for column names
    df = pd.DataFrame(tab[1:], columns=tab[0])

    # Display the table in Streamlit
    st.write("# Hypotheses Evaluation")
    st.table(df)

    

    # Adding Final Features
    st.subheader("Adding Final Features")
    st.write("""
    Finally, we include the "season_cos" and "day_of_week_cos" columns based on EDA findings, completing the feature selection process.
    """)

    # Columns to add
    feat_to_add = ['datetime', 'mw_energy_consumption']

    # Final selected features
    cols_selected_full = cols_selected.copy()
    cols_selected_full.extend(feat_to_add)
    st.write("Final Selected Features:", cols_selected_full)

    # End of Part 4: Feature Selection
    st.markdown("---")
    st.write("""
    Feature selection has refined the dataset, retaining only the most relevant features for modeling. 
    This process enhances the model's efficiency and predictive power.
    """)
    st.markdown("---")

    # Start of Part 5: Machine Learning Modeling
    st.title("Part 5: Machine Learning Modeling")

    st.write("""
    In this section, we will implement and evaluate machine learning models to predict energy consumption.
    We'll use a RandomForestRegressor and evaluate its performance using different metrics.
    """)

    # Define a function to calculate performance metrics
    def ml_error(model_name, y, yhat):
        mae = mean_absolute_error(y, yhat)
        mape = mean_absolute_percentage_error(y, yhat)
        rmse = np.sqrt(mean_squared_error(y, yhat))

        return pd.DataFrame({'Model Name': model_name,
                            'MAE': mae,
                            'MAPE': mape,
                            'RMSE': rmse}, index=[0])

    # Define a function for cross-validation
    def cross_validation(X_training, kfold, model_name, model, verbose=False):
        mae_list = []
        mape_list = []
        rmse_list = []

        for k in reversed(range(1, kfold + 1)):
            if verbose:
                print('\nKFold Number: {}'.format(k))
            # start and end date for validation
            validation_start_date_1 = X_training[X_training['electric_company'] == 1]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_2 = X_training[X_training['electric_company'] == 2]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_3 = X_training[X_training['electric_company'] == 3]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_4 = X_training[X_training['electric_company'] == 4]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_5 = X_training[X_training['electric_company'] == 5]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_6 = X_training[X_training['electric_company'] == 6]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_7 = X_training[X_training['electric_company'] == 7]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_8 = X_training[X_training['electric_company'] == 8]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_9 = X_training[X_training['electric_company'] == 9]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_10 = X_training[X_training['electric_company'] == 10]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_11 = X_training[X_training['electric_company'] == 11]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)
            validation_start_date_12 = X_training[X_training['electric_company'] == 12]['datetime'].max() - datetime.timedelta(days=k * 12 * 30)

            validation_end_date_1 = X_training[X_training['electric_company'] == 1]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_2 = X_training[X_training['electric_company'] == 2]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_3 = X_training[X_training['electric_company'] == 3]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_4 = X_training[X_training['electric_company'] == 4]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_5 = X_training[X_training['electric_company'] == 5]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_6 = X_training[X_training['electric_company'] == 6]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_7 = X_training[X_training['electric_company'] == 7]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_8 = X_training[X_training['electric_company'] == 8]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_9 = X_training[X_training['electric_company'] == 9]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_10 = X_training[X_training['electric_company'] == 10]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_11 = X_training[X_training['electric_company'] == 11]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)
            validation_end_date_12 = X_training[X_training['electric_company'] == 12]['datetime'].max() - datetime.timedelta(days=(k - 1) * 12 * 30)

            # filtering dataset
            training = X_training[
                ((X_training['electric_company'] == 1) & (X_training['datetime'] < validation_start_date_1))
                | ((X_training['electric_company'] == 2) & (X_training['datetime'] < validation_start_date_2))
                | ((X_training['electric_company'] == 3) & (X_training['datetime'] < validation_start_date_3))
                | ((X_training['electric_company'] == 4) & (X_training['datetime'] < validation_start_date_4))
                | ((X_training['electric_company'] == 5) & (X_training['datetime'] < validation_start_date_5))
                | ((X_training['electric_company'] == 6) & (X_training['datetime'] < validation_start_date_6))
                | ((X_training['electric_company'] == 7) & (X_training['datetime'] < validation_start_date_7))
                | ((X_training['electric_company'] == 8) & (X_training['datetime'] < validation_start_date_8))
                | ((X_training['electric_company'] == 9) & (X_training['datetime'] < validation_start_date_9))
                | ((X_training['electric_company'] == 10) & (X_training['datetime'] < validation_start_date_10))
                | ((X_training['electric_company'] == 11) & (X_training['datetime'] < validation_start_date_11))
                | ((X_training['electric_company'] == 12) & (X_training['datetime'] < validation_start_date_12))]

            validation = X_training[
                ((X_training['electric_company'] == 1) & (X_training['datetime'] >= validation_start_date_1) & (
                            X_training['datetime'] <= validation_end_date_1))
                | ((X_training['electric_company'] == 2) & (X_training['datetime'] >= validation_start_date_2) & (
                            X_training['datetime'] <= validation_end_date_2))
                | ((X_training['electric_company'] == 3) & (X_training['datetime'] >= validation_start_date_3) & (
                            X_training['datetime'] <= validation_end_date_3))
                | ((X_training['electric_company'] == 4) & (X_training['datetime'] >= validation_start_date_4) & (
                            X_training['datetime'] <= validation_end_date_4))
                | ((X_training['electric_company'] == 5) & (X_training['datetime'] >= validation_start_date_5) & (
                            X_training['datetime'] <= validation_end_date_5))
                | ((X_training['electric_company'] == 6) & (X_training['datetime'] >= validation_start_date_6) & (
                            X_training['datetime'] <= validation_end_date_6))
                | ((X_training['electric_company'] == 7) & (X_training['datetime'] >= validation_start_date_7) & (
                            X_training['datetime'] <= validation_end_date_7))
                | ((X_training['electric_company'] == 8) & (X_training['datetime'] >= validation_start_date_8) & (
                            X_training['datetime'] <= validation_end_date_8))
                | ((X_training['electric_company'] == 9) & (X_training['datetime'] >= validation_start_date_9) & (
                            X_training['datetime'] <= validation_end_date_9))
                | ((X_training['electric_company'] == 10) & (X_training['datetime'] >= validation_start_date_10) & (
                            X_training['datetime'] <= validation_end_date_10))
                | ((X_training['electric_company'] == 11) & (X_training['datetime'] >= validation_start_date_11) & (
                            X_training['datetime'] <= validation_end_date_11))
                | ((X_training['electric_company'] == 12) & (X_training['datetime'] >= validation_start_date_12) & (
                            X_training['datetime'] <= validation_end_date_12))]

            # training and validation dataset
            # training
            xtraining = training.drop(['datetime', 'mw_energy_consumption'], axis=1)
            ytraining = training['mw_energy_consumption']

            # validation
            xvalidation = validation.drop(['datetime', 'mw_energy_consumption'], axis=1)
            yvalidation = validation['mw_energy_consumption']

            # model
            m = model.fit(xtraining, ytraining)

            # prediction
            yhat = m.predict(xvalidation)

            # performance
            m_result = ml_error(model_name, np.expm1(yvalidation), np.expm1(yhat))

            # store performance of each kfold interation
            mae_list.append(m_result['MAE'])
            mape_list.append(m_result['MAPE'])
            rmse_list.append(m_result['RMSE'])

        return pd.DataFrame({'Model Name': model_name,
                            'MAE CV': np.round(np.mean(mae_list), 2).astype(str) + ' +/- ' + np.round(np.std(mae_list), 2).astype(str),
                            'MAPE CV': np.round(np.mean(mape_list), 2).astype(str) + ' +/- ' + np.round(np.std(mape_list), 2).astype(str),
                            'RMSE CV': np.round(np.mean(rmse_list), 2).astype(str) + ' +/- ' + np.round(np.std(rmse_list), 2).astype(str)}, index=[0])

    # Split the data into training and testing sets
    st.subheader("Split the Data into Training and Testing Sets")
    x_train = X_train[cols_selected]
    x_test = X_test[cols_selected]

    # Prepare the full training data for cross-validation
    X_training = X_train[cols_selected_full]

    st.write(f"x_train shape: {x_train.shape}")
    st.write(f"X_training shape: {X_training.shape}")

    # Explanation of Error Metrics
    st.subheader("Error Metrics Explanation")
    st.write("""
    Before implementing the machine learning models, let's discuss the error metrics used to evaluate model performance:
    - **MAE (Mean Absolute Error)**: The average of the absolute differences between actual and predicted values. It is straightforward to understand and report to business stakeholders.
    - **MAPE (Mean Absolute Percentage Error)**: The percentage representation of the MAE, indicating how much the error means in terms of the actual value.
    - **RMSE (Root Mean Square Error)**: The square root of the average of the squared differences between actual and predicted values. It is sensitive to outliers and often used by data scientists to assess model performance.
    """)

    ### Average Model
    st.subheader("Average Model")
    st.write("""
    The average model calculates the mean energy consumption for each electric company and uses it as a baseline for comparison with the machine learning models.
    """)

    # Copy test set to preserve original values
    aux1 = x_test.copy()
    aux1['mw_energy_consumption'] = y_test.copy()

    # Calculate the mean energy consumption for each electric company
    aux2 = aux1[['electric_company', 'mw_energy_consumption']].groupby('electric_company').mean().reset_index().rename(columns={'mw_energy_consumption': 'predictions'})

    # Merge the mean predictions with the test set
    aux1 = pd.merge(aux1, aux2, how='left', on='electric_company')
    yhat_baseline = aux1['predictions']

    # -----------------------------------------------------------------------
    # Check for infinity or NaN values and replace them
    y_test_finite = np.isfinite(y_test)
    yhat_baseline_finite = np.isfinite(yhat_baseline)

    # Handle any infinite or NaN values
    if not np.all(y_test_finite):
        st.write("Warning: y_test contains infinite or NaN values. Handling them now.")
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=np.max(y_test_finite), neginf=np.min(y_test_finite))

    if not np.all(yhat_baseline_finite):
        st.write("Warning: yhat_baseline contains infinite or NaN values. Handling them now.")
        yhat_baseline = np.nan_to_num(yhat_baseline, nan=0.0, posinf=np.max(yhat_baseline_finite), neginf=np.min(yhat_baseline_finite))

    # Calculate performance metrics for the Average Model
    try:
        baseline_result = ml_error('Average Model', np.expm1(y_test), np.expm1(yhat_baseline))
        st.write("Performance Metrics for Average Model:")
        st.dataframe(baseline_result)
    except ValueError as e:
        st.error(f"Error in calculating performance metrics: {e}")

    # Show x_test for debugging purposes
    st.write("x_test sample:")
    st.write(x_test.head())

    # -----------------------------------------------------------------------

    # Calculate performance metrics for the Average Model
    #baseline_result = ml_error('Average Model', np.expm1(y_test), np.expm1(yhat_baseline))
    #st.write("Performance Metrics for Average Model:")
    #st.write(baseline_result)

    # Random Forest Model
    st.subheader("Random Forest Model")
    st.write("""
    Next, we'll implement the RandomForestRegressor with tuned hyperparameters.
    """)

    # Set parameters
    param_tuned = {
            'bootstrap': True,
            'max_depth': 60,
            'max_features': 'sqrt',
            'min_samples_leaf': 4,
            'min_samples_split': 10,
            'n_estimators': 800
            }

    # Model
    rf_tuned = RandomForestRegressor(bootstrap=param_tuned['bootstrap'],
                                      max_depth=param_tuned['max_depth'],
                                      max_features=param_tuned['max_features'],
                                      min_samples_leaf=param_tuned['min_samples_leaf'],
                                      min_samples_split=param_tuned['min_samples_split'],
                                      n_estimators=param_tuned['n_estimators']).fit(x_train, y_train)

    # Prediction
    yhat_rf_tuned = rf_tuned.predict(x_test)

    # Performance
    rf_result_tuned = ml_error('Random Forest Regressor', np.expm1(y_test), np.expm1(yhat_rf_tuned))
    st.write("Performance Metrics for Random Forest Regressor:")
    st.write(rf_result_tuned)

    # End of Part 5: Machine Learning Modeling
    st.markdown("---")

    # Start of Part 6: Translation and Interpretation of the Error
    st.markdown("---")
    st.title("Part 6: Translation and Interpretation of the Error")

    # Assuming df9, df92, and other dataframes are already defined

    st.write("""
    In this section, we translate the mathematical predictions and their errors into business insights, interpreting what the error metrics mean for each electric company over the next year.
    """)

    st.subheader("Rescaling Predictions")
    st.write("""
    The predictions and their errors are rescaled from logarithmic values to their original scale, providing more interpretable results.
    """)

    # Rescale
    df9 = X_test[cols_selected_full].copy()  # Make a copy to avoid the warning
    df9.loc[:, 'mw_energy_consumption'] = np.expm1(df9['mw_energy_consumption'])
    df9.loc[:, 'predictions'] = np.expm1(yhat_rf_tuned)

    # Display the rescaled data
    st.dataframe(df9.head())

    st.subheader("Business Performance")
    st.write("""
    We translate the predictions into business outcomes by calculating the expected energy consumption for each electric company, considering best and worst-case scenarios based on the error metrics.
    """)

    # Sum of predictions
    df91 = df9[['electric_company', 'predictions']].groupby('electric_company').sum().reset_index()

    # MAE and MAPE
    df9_aux1 = df9[['electric_company', 'mw_energy_consumption', 'predictions']].groupby('electric_company').apply(lambda x: mean_absolute_error(x['mw_energy_consumption'], x['predictions'])).reset_index().rename(columns={0: 'MAE'})
    df9_aux2 = df9[['electric_company', 'mw_energy_consumption', 'predictions']].groupby('electric_company').apply(lambda x: mean_absolute_percentage_error(x['mw_energy_consumption'], x['predictions'])).reset_index().rename(columns={0: 'MAPE'})

    # Merge
    df9_aux3 = pd.merge(df9_aux1, df9_aux2, how='inner', on='electric_company')
    df92 = pd.merge(df91, df9_aux3, how='inner', on='electric_company')

    # Scenarios
    df92['worst_scenario'] = df92['predictions'] - df92['MAE']
    df92['best_scenario'] = df92['predictions'] + df92['MAPE']

    # Order columns
    df92 = df92[['electric_company', 'predictions', 'worst_scenario', 'best_scenario', 'MAE', 'MAPE']]

    # Display business performance data
    st.dataframe(df92.head(12))

    st.write("""
    The table above shows the predicted energy consumption for each electric company along with the worst and best-case scenarios. This helps in understanding the potential range of energy usage and planning accordingly.
    """)

    st.subheader("MAPE Error Distribution")
    st.write("""
    The Mean Absolute Percentage Error (MAPE) is plotted to visualize the error distribution across different electric companies.
    """)

    # Plot MAPE distribution
    sns.scatterplot(x='electric_company', y='MAPE', data=df92)
    plt.title('MAPE Error Distribution by Electric Company')
    st.pyplot(plt)

    st.subheader("Total Performance")
    st.write("""
    The total performance of the model is assessed by summing the predictions and their respective worst and best-case scenarios across all companies.
    """)

    df93 = df92[['predictions', 'worst_scenario', 'best_scenario']].apply(lambda x: np.sum(x), axis=0).reset_index().rename(columns={'index': 'Scenario', 0: 'Values'})
    df93['Values'] = df93['Values'].map('{:,.2f} MW'.format)

    # Display total performance
    st.dataframe(df93)

    st.subheader("Machine Learning Performance")
    st.write("""
    The model's performance is evaluated by comparing actual energy consumption against predictions and visualizing the error distribution.
    """)

    df9['error'] = df9['mw_energy_consumption'] - df9['predictions']
    df9['error_rate'] = df9['predictions'] / df9['mw_energy_consumption']

    # Separate dataframes for visualization
    df9_1 = df9[(df9['electric_company'] == 1) | (df9['electric_company'] == 2) | (df9['electric_company'] == 3) | (df9['electric_company'] == 4) | (df9['electric_company'] == 5) | (df9['electric_company'] == 6)
                | (df9['electric_company'] == 7) | (df9['electric_company'] == 8) | (df9['electric_company'] == 11) | (df9['electric_company'] == 12)]
    df9_2 = df9[df9['electric_company'] == 9]
    df9_3 = df9[df9['electric_company'] == 10]

    # Plot energy consumption and predictions
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))

    sns.lineplot(ax=axs[0, 0], x='datetime', y='mw_energy_consumption', data=df9_1, label='ENERGY CONSUMPTION')
    sns.lineplot(ax=axs[0, 0], x='datetime', y='predictions', data=df9_1, label='PREDICTIONS')
    axs[0, 0].set_title('Energy Consumption vs Predictions (df9_1)')

    sns.lineplot(ax=axs[0, 1], x='datetime', y='mw_energy_consumption', data=df9_2, label='ENERGY CONSUMPTION')
    sns.lineplot(ax=axs[0, 1], x='datetime', y='predictions', data=df9_2, label='PREDICTIONS')
    axs[0, 1].set_title('Energy Consumption vs Predictions (df9_2)')

    sns.lineplot(ax=axs[1, 0], x='datetime', y='mw_energy_consumption', data=df9_3, label='ENERGY CONSUMPTION')
    sns.lineplot(ax=axs[1, 0], x='datetime', y='predictions', data=df9_3, label='PREDICTIONS')
    axs[1, 0].set_title('Energy Consumption vs Predictions (df9_3)')

    # Plot error rate
    sns.lineplot(ax=axs[1, 1], x='datetime', y='error_rate', data=df9_1)
    axs[1, 1].axhline(1, linestyle='--', color='red')
    axs[1, 1].set_title('Error Rate (df9_1)')

    sns.lineplot(ax=axs[2, 0], x='datetime', y='error_rate', data=df9_2)
    axs[2, 0].axhline(1, linestyle='--', color='red')
    axs[2, 0].set_title('Error Rate (df9_2)')

    sns.lineplot(ax=axs[2, 1], x='datetime', y='error_rate', data=df9_3)
    axs[2, 1].axhline(1, linestyle='--', color='red')
    axs[2, 1].set_title('Error Rate (df9_3)')

    # Plot error distribution
    sns.histplot(ax=axs[3, 0], x=df9['error'], bins=30, kde=True)
    axs[3, 0].set_title('Distribution of Errors')

    sns.scatterplot(ax=axs[3, 1], x=df9['predictions'], y=df9['error'])
    axs[3, 1].set_title('Predictions vs Errors')

    plt.tight_layout()
    st.pyplot(fig)

    st.write("""
    The plots above show how well the model performs by comparing actual vs. predicted values and visualizing the error rates and distributions.
    """)

    # End of Part 6: Translation and Interpretation of the Error
    st.markdown("---")

else:
    st.write("Please upload a ZIP file containing the energy data.")