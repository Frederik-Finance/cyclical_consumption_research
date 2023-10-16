import os
import datetime
import math
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from scipy import optimize, stats
from scipy.stats import t, shapiro, norm
from scipy.stats.mstats import winsorize
from sklearn.metrics import r2_score, mean_squared_error
from pandas.tseries.offsets import QuarterEnd
from regression.linear_regression import return_predictive_regression, two_state_predictive_regression
from independent.hamilton import hamiltons_method



class ForecastingModel:
    def __init__(self, dependent_data_file, independent_data_file):
        self.dependent_data = pd.read_csv(dependent_data_file)
        self.independent_data = pd.read_csv(independent_data_file)

    def calculate_cw_test_statistic(self, errors_restricted, errors_unrestricted):
        """
        Calculate the Clark and West (2007) test statistic.

        Parameters:
        errors_restricted (array-like): The forecast errors of the restricted model.
        errors_unrestricted (array-like): The forecast errors of the unrestricted model.

        Returns:
        cw_stat (float): The Clark and West (2007) test statistic.
        p_value (float): The p-value associated with the test statistic.
        """
        # Calculate the difference in squared forecast errors
        d = errors_restricted**2 - errors_unrestricted**2

        # Calculate the mean of these differences
        d_bar = np.mean(d)

        # Calculate the test statistic
        cw_stat = np.sqrt(len(d)) * d_bar

        # Calculate the p-value
        p_value = 1 - norm.cdf(cw_stat)

        return cw_stat, p_value

    def calculate_oos_f(self, u1, u2):
        """
        Calculate the OOS-F statistic.

        Parameters:
        u1 (array-like): The forecast errors of the first model. unrestricted
        u2 (array-like): The forecast errors of the second model.

        Returns:
        oos_f (float): The OOS-F statistic.
        """
        # Calculate OOS-F statistic
        oos_f = np.sum((u1**2 - u2**2)) / np.sum(u2**2)

        return oos_f

    def enc_new(self, u1, u2):
        """
        Calculate the ENC-NEW test statistic.

        Parameters:
        u1, u2 : array-like
            Arrays of forecast errors from models 1 and 2, respectively.

        Returns:
        float
            The ENC-NEW test statistic.
        """
        # Ensure inputs are numpy arrays
        u1 = np.array(u1)
        u2 = np.array(u2)

        # Calculate the numerator and denominator of the ENC-NEW statistic
        numerator = np.sum(u1 * (u1 - u2))
        denominator = np.sum(u2**2)

        # Calculate the ENC-NEW statistic
        enc_new_stat = numerator / denominator

        return enc_new_stat

    def calculate_forecast_errors(self, df, date, next_date, predictions_unrestricted, predictions_restricted, return_col, forecast_quarters):
        '''
        This function calculates the forecast errors for the unrestricted and restricted models.
        '''

        try:
            actual = df.loc[date, f'rt,t+{forecast_quarters}_cumulative']
            error_unrestricted = actual - predictions_unrestricted[-1]
            error_restricted = actual - predictions_restricted[-1]

        except ValueError as e:
            print(f"Error occurred at date {next_date}: {e}")
            error_unrestricted = np.nan
            error_restricted = np.nan

        return error_unrestricted, error_restricted

    def prepare_test_train(self, df, start_date, end_date, return_col):
        '''
        This function prepares the test and train datasets based on the specified start and end dates.
        It also creates additional columns in the train dataset for each forecast horizon.
        For each 'cct' in the same row, it creates 'rt,t+h' for the specified forecast horizon 'h'.
        '''

        # Normalize the index and convert start_date and end_date to datetime
        df.index = pd.to_datetime(df.index)
        df.index = df.index.normalize()
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        forecast_horizons = [1, 2, 4, 8, 12, 16, 20]

        for h in forecast_horizons:
            col_name = f'rt,t+{h}_cumulative'
            # Calculate the sum of returns for next 'h' quarters
            df[col_name] = df[return_col].rolling(window=h).sum().shift(-h)

        # Define training period before start_date
        train = df[df.index <= start_date + pd.DateOffset(months=3)].copy()

        # Define test period between start_date and end_date
        test = df[(df.index >= start_date) & (df.index <= end_date)]

        # Drop rows that contain NaN in any of the 'rt,t+h' columns
        train.dropna()

        return train, test

    def recursive_forecasting(self, start_date, end_date, forecast_quarters, Lettau_Ludwigson=True):
        '''
        This function implements the recursive forecasting method for one quarter
        '''

        # Initialize placeholders for models and predictions
        models_unrestricted = []
        models_restricted = []
        predictions_unrestricted = []
        predictions_restricted = []
        errors_restricted = []
        errors_unrestricted = []

        # Normalize the index and convert start_date and end_date to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        self.dependent_data.index = pd.to_datetime(self.dependent_data.index)
        self.dependent_data.index = self.dependent_data.index.normalize()
        lower_limit = 0.03  # 5th percentile
        upper_limit = 0.03  # 95th percentile
        # Define training period before start_date

        initial_train, test = self.prepare_test_train(
            self.dependent_data, start_date, end_date, 'rt,t+1')

        # Number of lags to be used for constructing cc
        initialize_cc = True

        # Loop over the test period
        for date in test.index:
            next_date = date + pd.DateOffset(months=3)
            # Make a copy of the training data to avoid modifying it in-place
            train = initial_train.copy()

            # Recursively apply the Hamilton method to estimate the regressor cyclical consumption
            # using consumption data to estimate cct up until date
            cc_recursive = hamiltons_method(k=24, up_to_including_date=date)
            cc_recursive = cc_recursive.set_index('date')

            # If it's the first run, initialize the 'cyclical_consumption' column
            if initialize_cc:
                train['cyclical_consumption'] = cc_recursive['cyclical_consumption']

            initialize_cc = False
        else:
            '''Restimating the full sample each time if set to False, conversely if Lettau_Ludwigson is True the values stay fixed'''
            if Lettau_Ludwigson == False:
                train.loc[:date, 'cyclical_consumption'] = cc_recursive.loc[:date, 'cyclical_consumption']

            # Add the latest datapoint
            train.at[date, 'cyclical_consumption'] = cc_recursive['cyclical_consumption'].iloc[-1]

        '''Fitting the models up to t-1 but not including t'''
        '''Selects cct-1-h-1'''
        adjust_fit_period = pd.DateOffset(months=3*forecast_quarters)

        X_unrestricted = sm.add_constant(
            train.loc[train.index < date - adjust_fit_period, 'cyclical_consumption'])
        y_unrestricted = train.loc[train.index <
                                    date - adjust_fit_period, f'rt,t+{forecast_quarters}_cumulative']
        model_unrestricted = sm.OLS(y_unrestricted, X_unrestricted).fit()
        models_unrestricted.append(model_unrestricted)

        # Fit the restricted model
        '''Selects cct-1-h-1'''
        X_restricted = sm.add_constant(
            pd.Series(1, index=train.loc[train.index < date - adjust_fit_period].index))
        y_restricted = train.loc[train.index <
                                 date - adjust_fit_period, f'rt,t+{forecast_quarters}_cumulative']
        model_restricted = sm.OLS(y_restricted, X_restricted).fit()
        models_restricted.append(model_restricted)
        '''using date to predict next_date, in other words using cct(date) to predict rt+h(next_date)'''

        if next_date in self.dependent_data.index:
            try:
                # Performing the forecast
                const_df = pd.DataFrame(1, index=[date], columns=['const'])

                exog_df = pd.concat([const_df, pd.DataFrame(
                    {'cyclical_consumption': train['cyclical_consumption'].loc[date]}, index=[date])], axis=1)
                predictions_unrestricted.append(
                    model_unrestricted.predict(exog_df)[0])

                predictions_restricted.append(
                    model_restricted.predict(sm.add_constant(pd.Series(1, index=[date])))[0])

                error_unrestricted, error_restricted = self.calculate_forecast_errors(
                    self.dependent_data, date, next_date, predictions_unrestricted, predictions_restricted, 'crspr_log_market_return', forecast_quarters)
                errors_unrestricted.append(error_unrestricted)
                errors_restricted.append(error_restricted)
                
            except ValueError as e:
                print(f"Error occurred at date {date}: {e}")
        else:
            print(
                f"Skipping date {next_date} as it's not in the original DataFrame.")

        fixed_consumption_series = train['cyclical_consumption']
        initial_train = self.dependent_data[self.dependent_data.index <= date].copy()  # Create a copy of the subset
        if Lettau_Ludwigson == True:
            initial_train.loc[:, 'cyclical_consumption'] = fixed_consumption_series

        # Define the lower and upper limits for winsorization
        errors_unrestricted = [x for x in errors_unrestricted if not math.isnan(x)]
        errors_restricted = [x for x in errors_restricted if not math.isnan(x)]

        # Winsorize the errors
        errors_unrestricted = winsorize(errors_unrestricted, limits=[lower_limit, upper_limit])
        errors_restricted = winsorize(errors_restricted, limits=[lower_limit, upper_limit])

        mse_unrestricted = np.mean(np.array(errors_unrestricted)**2)
        mse_restricted = np.mean(np.array(errors_restricted)**2)
        r2_oos = 1 - mse_unrestricted / mse_restricted

        K = 1
        # sample size
        T = len(initial_train)
        print(f'r2_oos {r2_oos} for quarter = {forecast_quarters} ')

        # Calculate the Clark and West (2007) test statistic and p-value
        cw_stat, p_value = self.calculate_cw_test_statistic(errors_restricted, errors_unrestricted)
        oos_f_stat = self.calculate_oos_f(errors_restricted, errors_unrestricted)
        enc_new_stat = self.enc_new(errors_restricted, errors_unrestricted)
        print('OOS-F statistic:', oos_f_stat)
        print('ENC-NEW statistic:', enc_new_stat)
        print('\n')

        return models_unrestricted, models_restricted, r2_oos, enc_new_stat, oos_f_stat

    def run_forecasting(self, start_dates, end_date, forecast_horizons):
        with pd.ExcelWriter('oos_results/results_table.xlsx') as writer:
            for start_date in start_dates:
                data = {'ENC-NEW': [], 'R2 OOS': [], 'OOS-F': []}
                for k in forecast_horizons:
                    _, _, r2_oos, enc_new_stat, oos_f_stat = self.recursive_forecasting(
                        start_date, end_date, k)
                    data['ENC-NEW'].append(enc_new_stat)
                    data['OOS-F'].append(r2_oos)
                    data['R2 OOS'].append(oos_f_stat)

                # Convert the dictionary to a DataFrame
                df_results = pd.DataFrame(data, index=forecast_horizons).T

                # Write each DataFrame to a different worksheet
                df_results.to_excel(
                    writer, sheet_name=f'Forecasting from {start_date[:4]}')

        print('Excel file has been written successfully.')
