import pandas as pd


def load(filepath):
    """
    function that takes in data from a csv and loads it into a dataframe
    also creates appropriate columns needed for analysis
    """

    data = pd.read_csv(filepath)
    
    #extracting year from date column as we may choose to use year as a predictand
    #to account for climate warming, or general decadal changes in weather patterns
    data['DATE'] = pd.to_datetime('DATE')
    data['YEAR'] = data.dt.year

    return data #returning loaded data as a geopandas data frame for analysis


def examine():
    """
    visualizes raw data before any analysis as important step prior to any analysis
    checks data from normalacy (add complete description here)
    """

def multivariate_regression():
    """
    carries out multivariate regression on the dataset
    """


def desicion_tree():
    """
    finds the optimal degree of k for clustering
    subsequently uses k to create a desicion tree of depth k
    to predict whether a certain day is picni-able
    """


def random_forest():
    """
    creates a random forest model to predict whether
    a certain day is picni-able
    """