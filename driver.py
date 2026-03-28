import utilities as util


def main(filepath):

    "Loading data..."
    data = util.load_data(filepath) #data is loaded, NA containing rows deleted and new columns created

    predictand_name = 'BUDAPEST_picnic_weather'

    #data is checked for balance between classes, and correlation between variables
    #predictand is seperated out
    percentage_true_data_points, data, predictand = util.preprocessing(data, predictand_name)
    recall_value = util.logistic_regression(data, predictand)
    
    if (percentage_true_data_points > 60 or percentage_true_data_points < 40) or recall_value < 0.7:
        data, predictand = util.smote(data, predictand) #synthetic data is created using smote algorithm to balance the classes
        

    # #decision_tree_model = desicion_tree(data)
    # #random_forest_model = random_forest(data)
    


main("data/Budapest_Picnic_Prediction.csv")


#1 - check data normalacy (like last term)
#2 - regression (multi-variate?)
#3 - finding best k value + decision tree
#4 - fit to a model (simple eg random forest model) (literature review)





