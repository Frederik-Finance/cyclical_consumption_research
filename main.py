from classes.PredictiveRegression import PredictiveRegression
from classes.RecursiveForecasting import RecursiveForecasting
from classes.data_preparation.PrepareDependentData import PrepareDependentData
from classes.data_preparation.PrepareIndependentData import PrepareIndependentData
from classes.data_preparation.PrepareRecessionData import PrepareRecessionData
from classes.data_preparation.PrepareDependentData import PrepareDependentData
import pandas as pd





def main():

    # Create the dependent and independent data for all variables
    dependent_data = PrepareDependentData()
    independent_data = PrepareIndependentData()
    recession_data = PrepareRecessionData()

    # Create instances of the classes with the data as arguments
    predictive_regression = PredictiveRegression(dependent_data, independent_data)
    recursive_forecasting = RecursiveForecasting(dependent_data, independent_data)
    prepare_dependent_data = PrepareDependentData(dependent_data)

    # Use the analysis classes to perform the analysis

# Ensure the main function is called when the script is run directly
if __name__ == "__main__":
    main()
