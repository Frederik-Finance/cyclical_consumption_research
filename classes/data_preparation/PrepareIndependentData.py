import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from scipy import stats
from matplotlib.ticker import MaxNLocator




# Class to prepare the data
class PrepareIndependentData:
    def __init__(self, weighted_returns_file, consumption_data_file):
        self.weighted_returns = pd.read_csv(weighted_returns_file)
        self.consumption_data = pd.read_excel(consumption_data_file)

    # Hamilton's method to calculate cyclical consumption
    def hamiltons_method(self, k=8):
        print(f'Using k = {k}')

        # Prepare data
        data = self.consumption_data
        data['date'] = pd.date_range(start='1959-12-01', periods=len(data), freq='Q')
        data = data[data['date'].dt.year <= 2017]
        data['log_ct'] = np.log(data['Aggregate_Consumption'])

        for i in range(1, k + 4):
            data[f'log_ct_lag_{i}'] = data['log_ct'].shift(i)

        data = data.dropna()

        # Fit model
        X = data[[f'log_ct_lag_{i}' for i in range(k, k-4, -1)]]
        X = sm.add_constant(X)
        y = data['log_ct']

        model = sm.OLS(y, X).fit()

        data['cyclical_consumption'] = y - model.predict(X)

        print(model.summary())

        data['date'] = pd.to_datetime(data['date'])
        data['date'] = data['date'].dt.to_period('M').dt.to_timestamp()

        data[['date', 'cyclical_consumption']].to_csv('./data/out/cc.csv', index=False)

        return data['cyclical_consumption'], pd.to_datetime(data.date)



# Method to plot the cyclical consumption
def plot_cyclical_consumption(residuals, dates):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, residuals, color='darkblue')  # Updated line color
    ax.set_xlabel('Time')
    ax.set_ylabel('Cyclical Consumption')
    ax.set_title('Detrended Cyclical Consumption Europe')

    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.yaxis.set_ticks(np.arange(min(residuals), max(residuals), 0.02))

    plt.setp(ax.get_xticklabels(), rotation=45)

    plt.savefig('cyclical_consumption.png', dpi=300, bbox_inches='tight')

# Method to calculate the statistics
def calculate_statistics(residuals, dataset = 'Eurozone'):

    print(f'Interpretation stats for the {dataset} ')
    # Unconditional mean
    mean = np.mean(residuals)
    print(f"Unconditional mean: {mean}")

    # Standard deviation
    std_dev = np.std(residuals)
    print(f"Standard deviation: {std_dev}")

    # First-order autocorrelation
    autocorr = pd.Series(residuals).autocorr(lag=1)
    print(f"First-order autocorrelation: {autocorr}")

    # Skewness
    skewness = stats.skew(residuals)
    print(f"Skewness: {skewness}")

    # Kurtosis
    kurtosis = stats.kurtosis(residuals)
    print(f"Kurtosis: {kurtosis}")

# Method to plot cyclical consumption with recession indicators
def plot_cyclical_consumption_with_rec(df):
    # Convert the index to datetime if it's not already
    df = df.copy()
    df = df.set_index('date', drop=True)
    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['cyclical_consumption'], color='black')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cyclical Consumption')
    ax.set_title('Cyclical Consumption Over Time')

    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Modify y-axis ticks to have intervals of 0.02
    ax.yaxis.set_ticks(np.arange(min(df['cyclical_consumption']), max(df['cyclical_consumption']), 0.02))

    plt.setp(ax.get_xticklabels(), rotation=45)

    recession_start = None
    for i in range(len(df)):
        if df['USREC'].iloc[i] == 1:
            if recession_start is None:  # Start of recession
                recession_start = df.index[i]
        else:
            if recession_start is not None:  # End of recession
                ax.axvspan(recession_start, df.index[i], facecolor='gray', alpha=0.3)
                recession_start = None

    # Handle case where recession doesn't end
    if recession_start is not None:
        ax.axvspan(recession_start, df.index[-1], facecolor='gray', alpha=0.3)

    plt.savefig('cyclical_consumption.png', dpi=300, bbox_inches='tight')
    plt.show()
