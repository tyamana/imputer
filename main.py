from imputers import Imputer
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

__author__ = 'Artem Zhirokhov'


#"""
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

def neighbors(X, y, results, method_name, min_neigh_num, max_neigh_num):
    max_acc = 0
    best_neigh = 0
    epsilon = 0.005

    for neighbors_num in range(min_neigh_num, max_neigh_num):
        knn = KNC(metric='manhattan', n_neighbors=neighbors_num)

        X_scaled = StandardScaler().fit_transform(X)
        acc_score = cross_val_score(knn, X_scaled, y, cv=5)
        mean_acc_score = acc_score.mean()

        if mean_acc_score > max_acc + epsilon:
            max_acc = mean_acc_score
            best_neigh = neighbors_num

    results[method_name] = (max_acc, best_neigh)


data = get_data('mammographic.dat')

columns = list(data.columns.values)
X_columns = columns[:-1]
y_column = columns[-1]
X = data[X_columns]
Y = data[y_column]

results = {}

neighbors(Imputer(strategy='mean').fit_transform(X), Y, results, 'mean', 10, 70)
neighbors(Imputer(strategy='class_median').fit_transform(X, Y), Y, results, 'class_median', 10, 70)
neighbors(Imputer(strategy='class_mean').fit_transform(X, Y), Y, results, 'class_mean', 10, 70)
neighbors(Imputer(strategy='knn').fit_transform(X), Y, results, 'knn', 10, 70)

import operator
results_sorted_by_maxAcc = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
for item in results_sorted_by_maxAcc:
    print(item[1], item[0])
#"""

#imputer = Imputer(strategy='knn', verbose=True)
#imputer.fit(X)

#x = Imputer(strategy='class_mean', verbose=True)
#df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, np.nan, 7], 'B': [3, 4, 5, 6, 7, 8, 9, np.nan]})
#df = pd.DataFrame({'A': ['a', 'a', 'a', 'b', 'b', 'b', np.nan, np.nan], 'B': [3, 4, 5, 6, 7, 8, np.nan, np.nan]})
#print(df)
#y1 = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
#print(x.fit_transform(df, y1))
#print(x.fit_transform(df, y))
#print(Imputer().fit_transform(df, y))