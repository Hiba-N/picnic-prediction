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
    data = pd.dropna(data)
    
    #extracting year from date column as we may choose to use year as a predictand
    #to account for climate warming, or general decadal changes in weather patterns
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['YEAR'] = data.dt.year

    return data #returning loaded data as a geopandas data frame for analysis

def data_counter(data):
    #check trues and falses
    total_data_points = data.count()
    true_data_points = data.count(data[data['BUDAPEST_picnic_weather'] == True])
    false_data_points = data.count(data[data['BUDAPEST_picnic_weather'] == False])

    print("The total amount of points is: ", total_data_points)
    print("The amount of true data points is:" , true_data_points)
    print("The amount of false data points is:" , false_data_points)

    percentage_true_data_points = (true_data_points/(total_data_points))*100

    return percentage_true_data_points

def preprocessing(data):
    """
    visualizes raw data before any analysis as important step prior to any analysis
    checks data from normalacy (add complete description here)
    """

    sns.pairplot(data, hue ='day')
    plt.show()

    percentage_true_data_points = data_counter(data)

    return percentage_true_data_points


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


def logistic_regression(data):
    """
    carries out logistic regression on the dataset
    """

    # logistic regression

    x = data[:, :-1]  # all columns except last
    y = data[:, -1]   # last column as target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=23)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test)) * 100
    print(f"Logistic Regression model accuracy:", accuracy)

    # Predict labels
    y_pred = model.predict(x_test)

    # Call confusion_matrix function
    acc, prec, rec, f1 = confusion_matrix(y_test, y_pred)

    return rec



def smote(data):
        #drop or feature engineer, smote

    old_percentage_true_data_points = data_counter(data)

    smote = SMOTE(sampling_strategy='minority', random_state=42)

    x = data[:, :-1]  # all columns except last
    y = data[:, -1]   # last column as target

    x_smoted, y_smoted = smote.fit_resample(x, y)

    data[:, :-1] = x_smoted
    data[:, -1]  = y_smoted

    new_percentage_true_data_points = data_counter(data)

    print("The old ratio of true/false was: ", old_percentage_true_data_points, ":", 100 - old_percentage_true_data_points)
    print("The new ratio of true/false is: ", new_percentage_true_data_points, ":", 100 - new_percentage_true_data_points)


def decision_tree():

def random_forest():
    """
    creates a random forest model to predict whether
    a certain day is picni-able
    """