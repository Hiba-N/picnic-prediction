import utilities as util

# the driver script should at least take one command line argument, 
# the filename which contains the data.


# defensive programming for the command line argument(s)

def main(filepath):

    data = util.load_data(filepath) #data is loaded, NA containing rows deleted and new columns created

    percentage_true_data_points = util.preprocessing(data) #data is checked for balance between classes, and correlation between variables
    recall_value = util.logistic_regression(data)
    
    if (percentage_true_data_points < 60 or percentage_true_data_points > 40) and recall_value < 0.7:
        data = util.smote(data) #synthetic data is created using smote algorithm to balance the classes

    decision_tree_model = desicion_tree(data)
    random_forest_model = random_forest(data)
    



# functions can not access variables that are not passed to them!
# you must perform at least four statistical or model-fitting techniques on the data, 
# each one in its own function. 

#1 - check data normalacy (like last term)
#2 - regression (multi-variate?)
#3 - finding best k value + decision tree
#4 - fit to a model (simple eg random forest model) (literature review)


# The function should report the results of the particular analysis:
    # probability/statistical estimators computations
    # model fitting
    # statistical hypothesis testing
    # statistical  power analysis
    # classification model


# . . .
# Additionally:

# you are welcome to include other statistical methods or machine learning algorithms
# that we haven't discussed in class, but will need to briefly explain them and why 
# you are using them.
# you may also include shell scripting, in case you need to handle several files.
# you may re-use no more than two types of analysis from previous assignments.
# You must submit:

# the git log for the repository you created,
# any data files used in the analysis,
# your main driver and utilities file,


# Submit your main driver script, utilities file, data files, your report in PDF format, and the output of "git log" from your assignment repository.

# To capture the output of 'git log' use redirection, git log > git.log, and hand in the "git.log" file.