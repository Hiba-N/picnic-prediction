import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from imblearn.over_sampling import SMOTE
import seaborn as sns


def load_data(filepath):
    """
    function that takes in data from a csv and loads it into a dataframe
    also creates appropriate columns needed for analysis
    """

    data = pd.read_csv(filepath)
    data = data.dropna()
    
    #extracting year from date column as we may choose to use year as a predictand
    #to account for climate warming, or general decadal changes in weather patterns
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['YEAR'] = data['DATE'].dt.year
    data.drop('DATE', axis=1, inplace=True)

    return data #returning loaded data as a geopandas data frame for analysis

def data_counter(data, predictand):
    """
    Counts total, True, and False points for a boolean column.
    Returns percentage of True values.
    """
    total_data_points = len(data)
    true_data_points = sum(predictand)
    false_data_points = total_data_points - true_data_points

    print("The total amount of points is: ", total_data_points)
    print("The amount of true data points is:", true_data_points)
    print("The amount of false data points is:", false_data_points)

    percentage_true_data_points = (true_data_points / total_data_points) * 100

    print("The percentage of true_data_points is: ", percentage_true_data_points)

    return percentage_true_data_points

def preprocessing(data, predictand_name):
    """
    Visualizes raw data before any analysis as an important step prior to analysis.
    Checks data for normality and relationships between variables.
    """
      
    # Separate target and features
    y = data[predictand_name]
    features = data.drop(columns=predictand_name)  # features only
 
    # print("Creating pairplot...")

    # sns.set_theme(style="ticks")  # nicer style
    # pairplot = sns.pairplot(
    #     pd.concat([features, y], axis=1),
    #     hue=predictand,
    #     palette='Set2',
    #     markers='o',
    #     plot_kws={'s': 20, 'alpha': 0.6},
    #     diag_kind='hist'
    # )

    # # Reduce font size of x and y labels
    # for ax in pairplot.axes.flatten():
    #     ax.xaxis.label.set_size(8)   # X-axis label
    #     ax.yaxis.label.set_size(8)   # Y-axis label
    #     ax.tick_params(axis='both', labelsize=6)  # tick labels
        
    # # Save figure
    # pairplot.savefig("figures/pairplot.png")
    # plt.close()
        
    # Compute percentage of True values in target
    percentage_true_data_points = data_counter(features, y)
    
    return percentage_true_data_points, data, y


def confusion_matrix(actual, predicted):

    acc = accuracy_score(actual, predicted)
    prec = precision_score(actual, predicted)
    rec = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)

    print(f"Accuracy : {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall   : {rec:.2f}")
    print(f"F1 Score : {f1:.2f}")

    return acc, prec, rec, f1


def logistic_regression(data, predictand):
    """
    carries out logistic regression on the dataset
    """

    # logistic regression
    x_train, x_test, y_train, y_test = train_test_split(data, predictand, test_size=0.20, random_state=23)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test)) * 100
    print(f"Logistic Regression model accuracy:", accuracy)

    # Predict labels
    y_pred = model.predict(x_test)

    # Call confusion_matrix function
    acc, prec, rec, f1 = confusion_matrix(y_test, y_pred)

    return rec



def smote(data, predictand):
        #drop or feature engineer, smote

    old_percentage_true_data_points = data_counter(data, predictand)

    smote = SMOTE(sampling_strategy='minority', random_state=42)
    x_smoted, y_smoted = smote.fit_resample(data, predictand)

    new_percentage_true_data_points = data_counter(x_smoted, y_smoted)

    print("The old ratio of true/false was: ", old_percentage_true_data_points, ":", 100 - old_percentage_true_data_points)
    print("The new ratio of true/false is: ", new_percentage_true_data_points, ":", 100 - new_percentage_true_data_points)

    return x_smoted, y_smoted


# def decision_tree():

# def random_forest():
#     """
#     creates a random forest model to predict whether
#     a certain day is picni-able
#     """