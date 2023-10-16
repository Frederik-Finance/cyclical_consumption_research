
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import groupby



class PrepareRecessionData:
    ALL_COUNTRIES = ['GBR', 'FRA', 'DEU']
    CC_DATA_PATH = Path(r'C:\Users\surface pro 7\desktop\bachelorarbeit\code\cyclical_consumption_analysis_europe\data\out\cc.csv')
    RECESS_DATA_BASE_PATH = Path(r'C:\Users\surface pro 7\desktop\bachelorarbeit\code\cyclical_consumption_analysis_europe\data\country_data\recession_data')

    def __init__(self, ceiling_threshold=0.9):
        self.ceiling_threshold = ceiling_threshold

    def load_data(self):
        # Load recession data
        self.recession_data = self.aggregate_recession_data()

        # Load cyclical consumption data and prepare it
        self.cc_data = pd.read_csv(self.CC_DATA_PATH)
        self.prepare_cyclical_consumption_data()

        # Combine the data
        self.data = pd.merge(self.cc_data, self.recession_data, how='left', on='date')

        self.data = self.data[self.data['date'] >= pd.Period('2000-01')]

    def aggregate_recession_data(self):
        aggregated_rec_df = pd.DataFrame()
        for country in self.ALL_COUNTRIES:
            country_rec_df = self.prepare_country_data(country)
            aggregated_rec_df = pd.merge(aggregated_rec_df, country_rec_df, on='date', how='outer') if not aggregated_rec_df.empty else country_rec_df
        aggregated_rec_df['EUROREC'] = (aggregated_rec_df[[f'{country}REC' for country in self.ALL_COUNTRIES]].sum(axis=1) >= self.ceiling_threshold * len(self.ALL_COUNTRIES)).astype(int)
        aggregated_rec_df.dropna(inplace=True)
        return aggregated_rec_df

    def prepare_country_data(self, country):
        path_str = self.RECESS_DATA_BASE_PATH / f'{country}REC.csv'
        country_rec_df = pd.read_csv(path_str)
        country_rec_df['date'] = (pd.to_datetime(country_rec_df['DATE']) + pd.DateOffset(months=2)).dt.to_period('M')
        country_rec_df[f'{country}REC'] = country_rec_df[f'{country}REC'].replace('.', np.nan).astype(float).apply(np.ceil).dropna().astype(int)
        country_rec_df.drop(columns=['DATE'], inplace=True)
        return country_rec_df

    def prepare_cyclical_consumption_data(self):
        self.cc_data['date'] = pd.to_datetime(self.cc_data['date']).dt.to_period('M')
        mean_cc = self.cc_data['cyclical_consumption'].mean()
        std_cc = self.cc_data['cyclical_consumption'].std()
        self.cc_data['cc_recession_indicator'] = (self.cc_data['cyclical_consumption'] < mean_cc - std_cc).astype(int)
        self.cc_data.drop(['cyclical_consumption', 'index'], axis=1, inplace=True)

    def print_sorted_average_recession_lengths(self):
        avg_lengths = {country: self.average_recession_length(self.data[f'{country}REC']) for country in self.ALL_COUNTRIES}
        avg_lengths['EUROREC'] = self.average_recession_length(self.data['EUROREC'])
        avg_lengths['cc_recession_indicator'] = self.average_recession_length(self.data['cc_recession_indicator'])
        sorted_countries = sorted(avg_lengths, key=avg_lengths.get)
        print("The sorted average recession lengths in quarters are:")
        for country in sorted_countries:
            print(f"{country}: {round(avg_lengths[country], 2)}")

    @staticmethod
    def average_recession_length(series):
        series = series.dropna().astype(int)
        lengths = [len(list(g)) for k, g in groupby(series) if k==1]
        return round(sum(lengths) / len(lengths), 2) if lengths else 0
