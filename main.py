from imputers import Imputer
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
import operator
import xgboost as xgb

__author__ = 'Artem Zhirokhov'



#from .dat to pandas.DataFrame
def get_data(filename):
    f = open(filename, 'r')
    array = []
    names_of_columns_was_read = False
    for line in f:
        values = line.split(',')
        values[-1] = values[-1][:-1]
        if names_of_columns_was_read is False:
            for i in range(1, len(values)):
                values[i] = values[i][1:]
            names_of_columns_was_read = True
        else:
            for i in range(len(values)):
                if values[i] == '?':
                    values[i] = np.nan
                elif '.' in values[i]:
                    values[i] = float(values[i])
                elif values[i].isdigit():
                    values[i] = int(values[i])
        array.append(values)

    data = pd.DataFrame(array[1:], columns=array[0])
    return data

def neighbors(X, y, results, method_name, min_neigh_num, max_neigh_num, step):
    max_acc = 0
    best_neigh = 0
    epsilon = 0.005

    for neighbors_num in range(min_neigh_num, max_neigh_num, step):
        knn = KNC(metric='manhattan', n_neighbors=neighbors_num)

        X_scaled = StandardScaler().fit_transform(X)
        acc_score = cross_val_score(knn, X_scaled, y, cv=5)
        mean_acc_score = acc_score.mean()

        if mean_acc_score > max_acc + epsilon:
            max_acc = mean_acc_score
            best_neigh = neighbors_num

    results[method_name] = (max_acc, best_neigh)

def getNumAndCatfeatures(X):
    num_columns = []
    cat_columns = []

    for column in X.columns:
        if type(X[column].dropna().iloc[0]) == str:
            cat_columns.append(column)
        else:
            num_columns.append(column)

    return (num_columns, cat_columns)


"""
data = get_data('mammographic.dat')

columns = list(data.columns.values)
X_columns = columns[:-1]
y_column = columns[-1]
X = data[X_columns]
Y = data[y_column]

#imptr = Imputer(strategy='xgboost', verbose=1).fit(X)
#imptr.transform(X, Y)
#imptr = Imputer(strategy='svm', verbose=1).fit(X)
#imptr.transform(X, Y)
#imptr = Imputer(strategy='knn', verbose=1).fit(X)
#imptr.transform(X, Y)
#imptr = Imputer(strategy='logistic_regr', verbose=1).fit(X)
#imptr.transform(X, Y)

results = {}

neighbors(Imputer(strategy='mean').fit_transform(X), Y, results, 'mean', 10, 70, 3)
neighbors(Imputer(strategy='class_median').fit_transform(X, Y), Y, results, 'class_median', 10, 70, 3)
neighbors(Imputer(strategy='class_mean').fit_transform(X, Y), Y, results, 'class_mean', 10, 70, 3)
neighbors(Imputer(strategy='knn').fit_transform(X), Y, results, 'knn', 10, 70, 3)
neighbors(Imputer(strategy='svm').fit_transform(X), Y, results, 'svm', 10, 70, 3)
neighbors(Imputer(strategy='xgboost').fit_transform(X), Y, results, 'xgboost', 10, 70, 3)
#neighbors(Imputer(strategy='logistic_regr').fit_transform(X), Y, results, 'lr', 10, 70, 3)

results_sorted_by_maxAcc = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
for item in results_sorted_by_maxAcc:
    print(item[1], item[0])

"""
data = pd.DataFrame().from_csv('train.csv')
data = data.sample(frac=0.01, random_state=410)

X = data[data.columns.values[1:]]
num_columns, _ = getNumAndCatfeatures(data)
X = data[num_columns]
Y = data[data.columns.values[0]]

#Imputer(strategy='xgboost', verbose=1).fit(X)

results = {}

neighbors(Imputer(strategy='xgboost').fit_transform(X), Y, results, 'xgboost', 10, 70, 3)
neighbors(Imputer(strategy='mean').fit_transform(X), Y, results, 'mean', 10, 70, 3)
neighbors(Imputer(strategy='class_median').fit_transform(X, Y), Y, results, 'class_median', 10, 70, 3)
neighbors(Imputer(strategy='class_mean').fit_transform(X, Y), Y, results, 'class_mean', 10, 70, 3)
#neighbors(Imputer(strategy='knn').fit_transform(X), Y, results, 'knn', 10, 70, 3)
neighbors(Imputer(strategy='svm').fit_transform(X), Y, results, 'svm', 10, 70, 3)
#neighbors(Imputer(strategy='logistic_regr').fit_transform(X), Y, results, 'lr', 10, 70, 3)

results_sorted_by_maxAcc = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
for item in results_sorted_by_maxAcc:
    print(item[1], item[0])

