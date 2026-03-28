import utilities as util


def main(filepath):

    "Loading data..."
    data = util.load_data(filepath) #data is loaded, NA containing rows deleted and new columns created

    predictand_name = 'picnic_weather'

    #data is checked for balance between classes, and correlation between variables
    #predictand is seperated out
    percentage_true_data_points, data, predictand = util.preprocessing(data, predictand_name) #bimary class count is calculated
    logistic_model, recall_value = util.logistic_regression(data, predictand) #logistic regression
    
    if (percentage_true_data_points > 60 or percentage_true_data_points < 40) or recall_value < 0.7: #checking whether classes need to be balanced
        data, predictand = util.smote(data, predictand) #synthetic data is created using smote algorithm to balance the classes
        

    decision_tree_model, acc = util.decision_tree(data, predictand) #binary desicion tree is modelled
    
    print("A prediction accuracy of ", acc, "has been achieved.")
    
main("data/Picnic_Predictions.csv")






