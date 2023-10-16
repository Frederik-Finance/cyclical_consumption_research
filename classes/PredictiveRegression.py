
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col


class AdvancedStatistics:
    def __init__(self, data):
        self.data = data

    def indep_std(self, cyclical_consumption_col):
        return self.data[cyclical_consumption_col].std()
    
    def Compute_IVX_Wald(self, y, X, K, M_n, beta):
        # OLS estimation of predictive regression system
        model = sm.OLS(y[1:], sm.add_constant(X[:-1]))
        results = model.fit()
        epsilon_hat = results.resid

        T, r = X.shape
        U_hat = np.empty((T-1, r))

        for j in range(r):
            model_j = sm.OLS(X[1:, j], sm.add_constant(X[:-1, j]))
            results_j = model_j.fit()
            U_hat[:, j] = results_j.resid

        # Compute short-run/long-run covariance matrices
        Sigma_hat_ee = (1 / len(epsilon_hat)) * (epsilon_hat @ epsilon_hat)
        Sigma_hat_eu = (1 / len(epsilon_hat)) * (epsilon_hat @ U_hat)
        Sigma_hat_uu = (1 / len(epsilon_hat)) * (U_hat.T @ U_hat)
        Omega_hat_uu = Sigma_hat_uu
        Omega_hat_eu = Sigma_hat_eu

        if M_n > 0:
            Lambda_hat_uu = np.zeros((r, r))
            Lambda_hat_ue = np.zeros((r, 1))

            for h in range(1, M_n+1):
                Lambda_hat_uu += (1 - h / (M_n + 1)) * (1 / (len(U_hat) - 1)) * (U_hat[h:].T @ U_hat[:-h])
                Lambda_hat_ue += (1 - h / (M_n + 1)) * (1 / (len(U_hat) - 1)) * (U_hat[h:].T @ epsilon_hat[:-h].reshape(-1, 1))

            Omega_hat_uu = Sigma_hat_uu + Lambda_hat_uu + Lambda_hat_uu.T
            Omega_hat_eu = Sigma_hat_eu + Lambda_hat_ue.T

        # Construct instruments
        R_nz = 1 - (1 / (len(y) - 1)) ** beta
        d_X = np.vstack((np.zeros((1, r)), np.diff(X, axis=0)))
        Z_tilde = np.zeros((T, r))

        for t in range(1, T):
            Z_tilde[t] = R_nz * Z_tilde[t - 1] + d_X[t]

        # Construct cumulative variables
        y_K = np.array([np.sum(y[t:t+K]) for t in range(len(y) - K + 1)])
        X_K = np.array([np.sum(X[t:t+K], axis=0) for t in range(len(X) - K + 1)])
        Z_tilde_K = np.array([np.sum(Z_tilde[t:t+K], axis=0) for t in range(len(Z_tilde) - K + 1)])

        # Construct matrices for demeaned variables and instruments
        n_K = len(y_K) - 1
        y_bar_K = np.mean(y_K[1:])
        x_bar_K = np.mean(X_K[:-1], axis=0)
        z_tilde_bar_K = np.mean(Z_tilde_K[:-K], axis=0)
        Y_K_under = y_K[1:] - y_bar_K
        X_K_under = X_K[:-1] - np.outer(np.ones(n_K), x_bar_K)
        Z_tilde_K = Z_tilde_K[:-1]
        Z_tilde = Z_tilde[:-K]

        # IVX estimation of demeaned predictive regression
        A_tilde_IVX_K = np.linalg.inv(X_K_under.T @ Z_tilde) @ (Z_tilde.T @ Y_K_under)

        # Compute IVX-Wald statistic
        Omega_hat_FM = Sigma_hat_ee - Omega_hat_eu @ np.linalg.inv(Omega_hat_uu) @ Omega_hat_eu.T
        M_K = (Z_tilde_K.T @ Z_tilde_K) * Sigma_hat_ee - n_K * (z_tilde_bar_K @ z_tilde_bar_K.T) * Omega_hat_FM
        Q_H_K = np.linalg.inv(Z_tilde.T @ X_K_under) @ M_K @ np.linalg.inv(X_K_under.T @ Z_tilde)
        W_IVX_K = A_tilde_IVX_K.T @ np.linalg.inv(Q_H_K) @ A_tilde_IVX_K

        p_value = 1 - stats.chi2.cdf(W_IVX_K, r)

        return A_tilde_IVX_K, W_IVX_K, p_value

    def resample_residuals(self, y, X, epsilon_hat, B, seed=0):
        """
        Resamples residuals and returns a 2D numpy array.

        Parameters
        ----------
            y : array-like
                Dependent variable.
            X : array-like
                Independent variable(s).
            epsilon_hat : float
                Estimated residuals from initial regression model.
            B : int
                The number of bootstrap samples to generate.
            seed : int, optional
                Random seed for reproducibility.

        Returns
        -------
            u_star : array
                Array of resampled residuals.
        """
        np.random.seed(seed)
        u_star = np.empty((B, len(y)))
        for b in range(B):
            u_star_b = np.random.randn(len(y)) * epsilon_hat
            u_star[b, :] = u_star_b
        return u_star

    def compute_wild_bootstrap_p_values(self, y, X, epsilon_hat, B, h, original_results, seed=0):
        """
        Computes and returns p-values for the regression coefficients using a wild bootstrap method.

        Parameters
        ----------
            y : array-like
                Dependent variable.
            X : array-like
                Independent variable(s).
            epsilon_hat : float
                Estimated residuals from initial regression model.
            B : int
                The number of bootstrap samples to generate.
            h : int
                Number of periods ahead for which returns are being predicted.
            original_results : RegressionResults
                Results object for the initial regression model.
            seed : int, optional
                Random seed for reproducibility.

        Returns
        -------
            p_values_lower : array
                One-sided p-values for the null hypothesis that each coefficient equals zero.
            None
        """
        num_coeffs = X.shape[1]
        t_stats = np.empty((B, num_coeffs))
        u_star = self.resample_residuals(y, X, epsilon_hat, B, seed)
        for b in range(B):
            y_star_b = np.mean(y) + u_star[b, :] * epsilon_hat
            model_star_b = sm.OLS(y_star_b, X)
            results_star_b = model_star_b.fit(cov_type='HAC', cov_kwds={'maxlags': h})
            t_stats[b, :] = results_star_b.tvalues

        # Compute one-sided p-value for H0: β = 0 against HA: β < 0
        p_values_lower = np.sum(t_stats <= original_results.t_values.values.reshape(1, -1), axis=0) / B

        # Print information about the p-values
        print(f'Empirical wild-bootstrap p-values for {h}-quarter ahead returns')
        print(f"P-values for null hypothesis of no effect of cyclical consumption on future expected returns (H0: β = 0): {p_values_lower}\nAlternative Hypothesis that cyclical consumption has a negative effect")
        print("\n")

        return p_values_lower, None




class SimplePredictiveRegression(AdvancedStatistics):
    def __init__(self, data):
        super().__init__(data)

    def return_predictive_regression(self, h_values, cyclical_consumption_col, return_col):
        models = []
        epsilon_hat = []

        for h in h_values:
            data = self.data.copy()
            data['cc'] = data[cyclical_consumption_col].shift(1)
            data = data.dropna()

            data[f'h={h}'] = data[return_col].rolling(window=h).sum().shift(-h)
            data = data.dropna()

            X = sm.add_constant(data['cc'])
            y = data[f'h={h}']

            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': h})
            models.append(model)
            epsilon_hat.append(model.resid)

        res = summary_col(models, regressor_order=models[-1].params.index.tolist(), stars=True)

        latex_output = res.as_latex()
        with open(r"C:\path\to\output.tex", "w") as f:
            f.write(latex_output)

        return res, models, epsilon_hat




class TwoStatePredictiveRegression(AdvancedStatistics):
    def __init__(self, data):
        super().__init__(data)

    def two_state_predictive_regression(self, h_values, cyclical_consumption_col, return_col, recession_col):
        eurorec_avg_length = average_recession_length(self.data[recession_col])
        print(f'Average length of {recession_col} recession series:', eurorec_avg_length)
        models = []

        for h in h_values:
            temp_data = self.data.copy()

            temp_data['lagged_cyclical_consumption'] = temp_data[cyclical_consumption_col].shift(1)
            temp_data[f'h={h}'] = temp_data[return_col].rolling(window=h).sum().shift(-h + 1)

            # Assuming the recession_col indicates a "bad_state" with 1 and "good_state" with 0
            temp_data['bad_state_indicator'] = temp_data[recession_col]

            temp_data['bad_state'] = temp_data['bad_state_indicator'] * temp_data['lagged_cyclical_consumption']
            temp_data['good_state'] = (1 - temp_data['bad_state_indicator']) * temp_data['lagged_cyclical_consumption']

            temp_data = temp_data.dropna()

            X = sm.add_constant(temp_data[['bad_state', 'good_state']])
            y = temp_data[f'h={h}']
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': h})

            models.append(model)

        res = summary_col(models, regressor_order=models[-1].params.index.tolist(), stars=True)

        latex_output = res.as_latex()
       
        with open(r"C:\path\to\output.tex", "w") as f:
            f.write(latex_output)

        return res, models



class RegressionAnalysis:
    def __init__(self, dependent_data_path, independent_data_path, recession_data_path, h_values):
        self.dependent_data_path = dependent_data_path
        self.independent_data_path = independent_data_path
        self.recession_data_path = recession_data_path
        self.h_values = h_values

        self.dependent_data = None
        self.independent_data = None
        self.recession_data = None
        self.msci_regression_df = None
        self.msci_regression_long_period_df = None

    def load_data(self):
        # Load datasets
        self.dependent_data = pd.read_csv(self.dependent_data_path)
        self.independent_data = pd.read_csv(self.independent_data_path)
        self.recession_data = pd.read_csv(self.recession_data_path)

        # Convert the 'Date' column to datetime
        self.dependent_data['Date'] = pd.to_datetime(self.dependent_data['Date'])
        self.independent_data['Date'] = pd.to_datetime(self.independent_data['Date'])
        self.recession_data['Date'] = pd.to_datetime(self.recession_data['Date'])

        # Merge dataframes
        merged_data = pd.merge(self.dependent_data, self.independent_data, on='Date')
        self.merged_data = pd.merge(merged_data, self.recession_data, on='Date')

        # Sort dataframe by date
        self.merged_data.sort_values(by='Date', inplace=True)



    def save_regression_data(self, output_path_short, output_path_long):
        self.msci_regression_df.to_csv(output_path_short)
        self.msci_regression_long_period_df.to_csv(output_path_long)


    def run_regressions(self, cyclical_consumption_col, return_col, recession_col):
        simple_regressions = SimplePredictiveRegression(self.msci_regression_df)
        simple_results, simple_models, simple_epsilon_hat = simple_regressions.return_predictive_regression(self.h_values, cyclical_consumption_col, return_col)

        two_state_regressions = TwoStatePredictiveRegression(self.msci_regression_df)
        two_state_results, two_state_models = two_state_regressions.two_state_predictive_regression(self.h_values, cyclical_consumption_col, return_col, recession_col)

        return simple_results, simple_models, simple_epsilon_hat, two_state_results, two_state_models
