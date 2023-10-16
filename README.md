# Research Code Repository

Welcome! This repository is a curated collection of refactored code snippets taken from my extensive research carried out during my bachelor's degree studies. While these excerpts may not function in their entirety due to their isolation from the full codebase, they serve to inspire those intrigued by analytical research and as a portfolio for potential recruiters.

## Content Focus

The showcased code specifically focuses on analysis conducted for both the United States and Europe. Please note that certain sections such as data cleaning procedures and large datasets have been intentionally excluded due to their size.

For those interested in obtaining the complete, functioning codebase, which includes individual analyses for each European country, please feel free to reach out to me at frederik.woite@mannheim.mail.de.


## Main Content

### Regression Models

The code provided in this repository is primarily focused on regression models, specifically predictive regression models. These models are used to predict future outcomes based on historical data. The code is divided into two main sections: One State and Two State regression models.

#### One State Regression Model

The One State regression model is a simple predictive regression model. It takes a dataset and a set of parameters, and uses the Ordinary Least Squares (OLS) method to fit a regression model. The model is then used to predict future outcomes based on the historical data. The code also includes functionality for resampling residuals and computing wild bootstrap p-values, which are used to assess the statistical significance of the regression coefficients.

#### Two State Regression Model

The Two State regression model is a more complex predictive regression model. It takes into account two different states of the world, which could represent different economic conditions, for example. The model fits separate regression models for each state, and uses these models to make predictions about future outcomes. The code also includes functionality for resampling residuals and computing wild bootstrap p-values, similar to the One State model.




# Out of sample analysis
The following code is an implementation of a recursive forecasting model. It is designed to perform out-of-sample analysis on a dataset, using a variety of statistical tests to evaluate the performance of the model. The code is written in Python and uses libraries such as pandas, numpy, and statsmodels for data manipulation and statistical analysis.

The main class, RecursiveForecasting, takes two CSV files as input: one for the dependent data and one for the independent data. The class contains several methods for calculating various test statistics, preparing the training and testing datasets, and performing the recursive forecasting.

The recursive_forecasting method is the core of the class. It takes a start date, an end date, and a forecast horizon as input. It then performs the recursive forecasting for the specified period and forecast horizon. The method also calculates the mean squared error (MSE) for the unrestricted and restricted models, and the out-of-sample R-squared (R2 OOS).

The run_forecasting method is used to run the recursive forecasting for multiple start dates and forecast horizons. It writes the results to an Excel file, with each worksheet corresponding to a different start date.

This code is particularly useful for researchers and analysts who are interested in forecasting and model evaluation. It provides a robust framework for out-of-sample analysis, which is a crucial aspect of validating the predictive power of a model.


# Overview of the IVX Estimator

The IVX (Instrumental Variable X) estimator is a robust econometric inference tool used for predictive regression models, particularly in the field of financial econometrics. This technique was proposed by Kostakis, Magdalinos, and Stamatogiannis (2015) and it provides a means of overcoming the issues associated with conventional predictive regression methods, like endogeneity and persistence in predictor variables.

## Mathematical Perspective

The IVX estimator constructs a series of instruments (factors or variables that help model the behavior of other variables) using lagged changes in the predictor variables. These instruments are then used in an instrumental variables regression framework to provide robust inference on predictive ability. This estimator is particularly useful in financial econometrics where the predictor variables (e.g., dividend-price ratio, earnings-price ratio, etc.) are highly persistent and endogenous. 

The key mathematical expressions involved in the IVX estimation are:

1. **Instrument Construction**: The instruments are constructed using the following equation:

   \[
   Z_t = R_{nz} Z_{t-1} + \Delta X_t
   \]

   where \(Z_t\) is the constructed instrument, \(R_{nz}\) is a decaying parameter, \(Z_{t-1}\) is the lagged instrument, and \(\Delta X_t\) is the first difference of the predictor variable \(X_t\).

2. **IVX Estimation**: The IVX estimator is obtained by the following equation:

   \[
   \tilde{A}_{IVX-K} = (X_K' Z)^{-1} Z' Y_K
   \]

   where \(X_K\) and \(Y_K\) are the cumulative sums of the predictor and dependent variables respectively, and \(Z\) is the constructed instrument.

3. **IVX-Wald Statistic**: The IVX-Wald statistic is computed as follows:

   \[
   W_{IVX-K} = \tilde{A}_{IVX-K}' Q_{H-K}^{-1} \tilde{A}_{IVX-K}
   \]

   where \(Q_{H-K}\) is a form of long-run covariance matrix.

The p-value for the IVX-Wald statistic is then obtained from the chi-square distribution with degrees of freedom equal to the number of predictor variables.

## Code Perspective

In the Python function `Compute_IVX_Wald`, the above mathematical procedures are implemented as follows:

1. **Data Preparation**: The function accepts as input the observed returns `y`, the matrix of predictor observations `X`, the forecast horizon `K`, the bandwidth parameter `M_n`, and the scalar `beta` which is used in the construction of instruments. It then prepares the data and computes residuals from ordinary least squares (OLS) regression of `y` on `X`.

2. **Instrument Construction**: The function constructs the instruments using the formula mentioned above. The `R_nz` parameter is computed as `1 - (1 / (len(y) - 1)) ** beta`. The first difference of `X` is computed using `np.diff(X, axis=0)`. These quantities are used to update `Z_tilde` in a loop.

3. **IVX Estimation**: The function computes the cumulative sums of `y`, `X`, and `Z_tilde` to create `y_K`, `X_K`, and `Z_tilde_K`. These are then used to compute `A_tilde_IVX_K` using the formula described above.

4. **IVX-Wald Statistic Computation**: The function computes `W_IVX_K` and the corresponding `p_value` from the chi-square distribution.

The function finally returns the coefficient estimates `A_tilde_IVX_K`, the IVX-Wald statistic `W_IVX_K`, and the `p_value` for the IVX-Wald statistic.

The code makes heavy use of NumPy for matrix operations and relies on the `statsmodels` library for performing the initial OLS regressions. The `scipy.stats` module is used to compute the p-value from the chi-square distribution.

## References

Kostakis, A., Magdalinos, T., & Stamatogiannis, M. P. (2015). Robust econometric inference for stock return predictability. Review of Financial Studies, 28(5), 1506-1553.
