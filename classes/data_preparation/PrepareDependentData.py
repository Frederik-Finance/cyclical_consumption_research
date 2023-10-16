import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from pathlib import Path


class PrepareDepedendentData:

    def __init__(self, data_folder, output_folder, gdp_files, msci_file, inflation_file, tbill_file):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.gdp_files = gdp_files
        self.msci_file = msci_file
        self.inflation_file = inflation_file
        self.tbill_file = tbill_file


    def prepare_dataframe(self):
        df_all = pd.DataFrame()
        for file in self.gdp_files:
            df_gdp = pd.read_csv(os.path.join(self.data_folder, file))
            df_gdp.rename(columns={'GDP': file.replace('.csv', '') + '_GDP', 'DATE': 'Date'}, inplace=True)
            df_gdp = df_gdp[(df_gdp['Date'] >= '1999-01-01') & (df_gdp['Date'] < '2020-01-01')]
            if df_all.empty:
                df_all = df_gdp
            else:
                df_all = pd.merge(df_all, df_gdp, how='outer', on='Date')

        return df_all

    def load_returns_data(self):
        return pd.read_excel(os.path.join(self.data_folder, self.msci_file))

    def weighted_returns(self, df_all, df_returns):
        df_all['Total_GDP'] = df_all[[file.replace('.csv', '') + '_GDP' for file in self.gdp_files]].sum(axis=1)
        df_weighted_returns = df_all[['Date']].copy()

        for country in ['BELGIUM', 'FRANCE', 'GERMANY', 'ITALY', 'NETHERLANDS', 'SWEDEN', 'SWITZERLAND', 'UNITED KINGDOM']:
            gdp_file = country[:3].upper() + 'GDP.csv'
            df_weighted_returns[country] = round(df_all[gdp_file.replace('.csv', '') + '_GDP'] / df_all['Total_GDP'], 5)
            df_weighted_returns[country] *= df_returns[country]

        df_weighted_returns['mscirn'] = df_weighted_returns[['BELGIUM', 'FRANCE', 'GERMANY', 'ITALY', 'NETHERLANDS', 'SWEDEN', 'SWITZERLAND', 'UNITED KINGDOM']].sum(axis=1)
        df_weighted_returns = df_weighted_returns[(df_weighted_returns['Date'] >= '1999-01-01') & (df_weighted_returns['Date'] < '2020-01-01')]

        return df_weighted_returns

    def run_analysis(self):
        df_all = self.prepare_dataframe()
        df_returns = self.load_returns_data()
        df_weighted_returns = self.weighted_returns(df_all, df_returns)

        df_weighted_returns.to_csv(os.path.join(self.output_folder, 'weighted_returns.csv'), index=False)
        # more code can be added here for further analysis

