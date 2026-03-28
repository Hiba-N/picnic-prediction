import utilities as util
import argparse
import os
import sys


def main(filepath):

    if not os.path.exists(filepath): #checking to see if path exists
        raise argparse.ArgumentTypeError(f"File does not exist: {filepath}")

    if not filepath.endswith(".csv"): # checking to see if extension is right
        raise argparse.ArgumentTypeError("File must be a .csv file")
    
    print("Loading data...")
    data = util.load_data(filepath) #data is loaded, NA containing rows deleted and new columns created

    predictand_name = 'picnic_weather'

    #data is checked for balance between classes, and correlation between variables
    #predictand is seperated out

    print("Preprocessing data...")
    percentage_true_data_points, data, predictand = util.preprocessing(data, predictand_name) #binary class count is calculated
    
    print("Carrying out logistic regression...")
    logistic_model, recall_value = util.logistic_regression(data, predictand) #logistic regression
    

    if (percentage_true_data_points > 60 or percentage_true_data_points < 40) or recall_value < 0.7: #checking whether classes need to be balanced
        print("Creating synthetic data through smote...")
        data, predictand = util.smote(data, predictand) #synthetic data is created using smote algorithm to balance the classes
        
    print("Carrying out binary desicion tree modelling...")
    decision_tree_model, acc = util.decision_tree(data, predictand) #binary desicion tree is modelled
    
    print("A prediction accuracy of ", acc, "has been achieved.")


parser = argparse.ArgumentParser(description="carry out statistical and modelling tests on filepath given") #parser initialized
parser.add_argument("filepath",type=str ,help="path to the data file")#defining argument and arg type to catch
args = parser.parse_args()
print(args)
main(args.filepath)








