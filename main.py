from imputers import Imputer
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor as GBR
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
import operator
import xgboost as xgb
import random
from copy import copy
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cross_validation import train_test_split

import warnings
warnings.filterwarnings("ignore")

__author__ = 'Artem Zhirokhov'

"""
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
"""

def getNumAndCatfeatures(X):
    num_columns = []
    cat_columns = []

    for column in X.columns:
        if type(X[column].dropna().iloc[0]) == str:
            cat_columns.append(column)
        else:
            num_columns.append(column)

    return (num_columns, cat_columns)

def getMissingDataRate(df):

    N = df.shape[0] * df.shape[1]
    naValues = 0

    for label in df.columns:
        nanValInCol = df[label].isnull()
        if True in nanValInCol.value_counts().index:
            naValues += nanValInCol.value_counts()[True]

    return float(naValues) / N

def objectToGoodType(df):
    new_df = pd.DataFrame()
    for column in df.columns:
        new_df[column] = pd.Series([df[column].loc[i] for i in df.index], index=df.index)

    return new_df


"""
def SVMclassification(X, y, results, method_name, min_c, max_c, step):
    max_acc = 0
    best_c = 0
    epsilon = 0.0005

    c_range = [1e-5*10**i for i in range(7)]
    #for c in [min_c + step*i for i in range(int((max_c - min_c) / step))]:
    for c in c_range:

        scaled_X = StandardScaler().fit_transform(X)

        if (len(y.value_counts().index.values) == 2):
            svc_acc = cross_val_score(svm.SVC(kernel='rbf', C=c), scaled_X, y, cv=5).mean()
        else:
            svc_acc = cross_val_score(svm.SVC(decision_function_shape='ovo', C=c), scaled_X, y, cv=5).mean()

        mean_acc_score = svc_acc.mean()

        if mean_acc_score > max_acc + epsilon:
            max_acc = mean_acc_score
            best_c = c

    results[method_name] = (max_acc, best_c)

def random_forest_classification(X, y, results, method_name, min_trees_num, max_trees_num, step):

    max_acc_score = 0.0
    best_trees_num = min_trees_num
    epsilon = 0.005

    trees_numbers = [(min_trees_num + 3*i) for i in range(int((max_trees_num - min_trees_num)/3))]

    for trees_num in trees_numbers:

        rf = RandomForestClassifier(n_estimators=trees_num)

        X_scaled = StandardScaler().fit_transform(X)

        acc_score = cross_val_score(rf, X_scaled, y, cv=5)
        acc_score = acc_score.mean()

        if acc_score > max_acc_score + epsilon:
            best_trees_num = trees_num
            max_acc_score = acc_score

    results[method_name] = (max_acc_score, best_trees_num)

def SVMClassification(X, y, results, method_name, minc_c, max_c, step):

    if (len(y.value_counts().index.values) == 2):

        scaled_X = StandardScaler().fit_transform(X)

        svc_acc = cross_val_score(svm.SVC(kernel='rbf'), scaled_X, y, cv=5).mean()

        results[method_name] = svc_acc

    else:

        scaled_X = StandardScaler().fit_transform(X)

        svc_acc = cross_val_score(svm.SVC(decision_function_shape='ovo'), scaled_X, y, cv=5).mean()

        results[method_name] = svc_acc

def logreg_classification(X, y, results, method_name, min_c, max_c, step):
    max_acc = 0
    best_c = 0
    epsilon = 0.0005

    c_range = [1e-5*10**i for i in range(7)]
    #for c in [min_c + step*i for i in range(int((max_c - min_c) / step))]:
    for c in c_range:
        clf = LogisticRegression(random_state=510, C=c)

        X_scaled = StandardScaler().fit_transform(X)
        acc_score = cross_val_score(clf, X_scaled, y, cv=5)
        mean_acc_score = acc_score.mean()

        if mean_acc_score > max_acc + epsilon:
            max_acc = mean_acc_score
            best_c = c

    results[method_name] = (max_acc, best_c)

def neighbors_regression(X, y, results, method_name, min_neigh_num, max_neigh_num, step):
    max_r2 = -1
    best_neigh = 0
    epsilon = 0.0005

    for neighbors_num in range(min_neigh_num, max_neigh_num, step):
        knn = KNR(metric='manhattan', n_neighbors=neighbors_num)

        X_scaled = StandardScaler().fit_transform(X)
        r2_score = cross_val_score(knn, X_scaled, y, cv=5, scoring='r2')
        mean_r2_score = r2_score.mean()

        if mean_r2_score > max_r2 + epsilon:
            max_r2 = mean_r2_score
            best_neigh = neighbors_num

    results[method_name] = (max_r2, best_neigh)

def gb_regression(X, y, results, method_name, min_for_num, max_for_num, step):
    max_r2 = -1
    best_neigh = 0
    epsilon = 0.005
    times_best_val_did_not_change = 0

    for for_num in range(min_for_num, max_for_num, step):
        #if times_best_val_did_not_change >= 5:
            #times_best_val_did_not_change = 0
            #break

        rgr = GBR(n_estimators=for_num)

        X_scaled = StandardScaler().fit_transform(X)
        r2_score = cross_val_score(rgr, X_scaled, y, cv=5, scoring='r2')
        mean_r2_score = r2_score.mean()

        if mean_r2_score > max_r2 + epsilon:
            max_r2 = mean_r2_score
            best_neigh = for_num
            times_best_val_did_not_change = 0
        else:
            times_best_val_did_not_change += 1

    results[method_name] = (max_r2, best_neigh)

def ridge_regression(X, y, results, method_name, min_alpha, max_alpha, step):
    max_r2 = -1
    best_alpha = 0
    epsilon = 0.0005

    #alpha_range = [1e-5]
    alpha_range = [1e-5*10**i for i in range(7)]

    #for alpha in [min_alpha + step*i for i in range(int((max_alpha - min_alpha) / step + 1))]:
    for alpha in alpha_range:
        rgr = Ridge(random_state=510, alpha=alpha, normalize=True)
        #rgr = Ridge(random_state=510, alpha=alpha)

        X_scaled = StandardScaler().fit_transform(X)
        r2_score = cross_val_score(rgr, X_scaled, y, cv=5, scoring='r2')
        mean_r2_score = r2_score.mean()

        if mean_r2_score > max_r2 + epsilon:
            max_r2 = mean_r2_score
            best_alpha = alpha

    results[method_name] = (max_r2, best_alpha)

def neighbors_classification(X, y, results, method_name, min_neigh_num, max_neigh_num, step):
    max_acc = 0
    best_neigh = 0
    epsilon = 0.0005

    for neighbors_num in range(min_neigh_num, max_neigh_num, step):
        knn = KNC(metric='manhattan', n_neighbors=neighbors_num)

        X_scaled = StandardScaler().fit_transform(X)
        acc_score = cross_val_score(knn, X_scaled, y, cv=5)
        mean_acc_score = acc_score.mean()

        if mean_acc_score > max_acc + epsilon:
            max_acc = mean_acc_score
            best_neigh = neighbors_num

    results[method_name] = (max_acc, best_neigh)
"""

def gb_regression(X, y, results, method_name, min_for_num, max_for_num, step):
    max_r2 = -1
    best_neigh = 0
    epsilon = 0.005
    times_best_val_did_not_change = 0

    estimators_range = [50, 100, 150]

    #for for_num in range(min_for_num, max_for_num, step):
    for for_num in estimators_range:
        #if times_best_val_did_not_change >= 5:
            #times_best_val_did_not_change = 0
            #break

        rgr = GBR(n_estimators=for_num)

        X_scaled = StandardScaler().fit_transform(X)
        r2_score = cross_val_score(rgr, X_scaled, y, cv=5, scoring='r2')
        mean_r2_score = r2_score.mean()

        if mean_r2_score > max_r2 + epsilon:
            max_r2 = mean_r2_score
            best_neigh = for_num
            times_best_val_did_not_change = 0
        else:
            times_best_val_did_not_change += 1

    results[method_name] = (max_r2, best_neigh)

def data_processing(path, string_to_print):
    #touse
    data = pd.read_csv(path)
    data = data.drop('Unnamed: 0', 1)
    if 'bikesharing' in string_to_print:
        data = data.sample(frac=0.5, random_state=552)
    print(data.shape)
    Y = data[data.columns.values[-1]]
    X = data[data.columns.values[:-1]]

    print(getMissingDataRate(X))

    results = {}

    gb_regression(Imputer(strategy='mean').fit_transform(X), Y, results, 'mean', 3, 30, 3)
    gb_regression(Imputer(strategy='logistic_regr').fit_transform(X), Y, results, 'logistic_regr', 3, 30, 3)
    gb_regression(Imputer(strategy='knn').fit_transform(X), Y, results, 'knn', 3, 30, 3)
    gb_regression(Imputer(strategy='svm').fit_transform(X), Y, results, 'svm', 3, 30, 3)
    gb_regression(Imputer(strategy='xgboost').fit_transform(X), Y, results, 'xgboost', 3, 30, 3)
    gb_regression(Imputer(strategy='kmeans').fit_transform(X, **{'n_clusters': 5,
                                                                        'max_iter': 300,
                                                                        'n_init': 100,
                                                                        'init': 'k-means++',
                                                                        'n_jobs': 2,
                                                                        'random_state': 282}),
                         Y, results, 'kmeans5', 3, 30, 3)
    gb_regression(Imputer(strategy='random_forest').fit_transform(X), Y, results, 'random_forest', 3, 30, 3)
    #"""
    print(string_to_print)
    results_sorted_by_maxAcc = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
    for item in results_sorted_by_maxAcc:
        print(item[1], item[0])


data_processing('/home/tyamana/data_from_kaggle/bikesharing/data5.csv', '(bikesharing, 2722x11, gb, 5%):')
data_processing('/home/tyamana/data_from_kaggle/bikesharing/data15.csv', '(bikesharing, 2722x11, gb, 15%):')
data_processing('/home/tyamana/data_from_kaggle/bikesharing/data30.csv', '(bikesharing, 2722x11, gb, 30%):')
data_processing('/home/tyamana/data_from_kaggle/bikesharing/data50.csv', '(bikesharing, 2722x11, gb, 50%):')

data_processing('/home/tyamana/missing_values/datasets/casp/data15.csv', '(casp, 4573x10, gb, 15%):')
data_processing('/home/tyamana/missing_values/datasets/casp/data30.csv', '(casp, 4573x10, gb, 30%):')
data_processing('/home/tyamana/missing_values/datasets/casp/data50.csv', '(casp, 4573x10, gb, 50%):')

data_processing('/home/tyamana/missing_values/datasets/CCPP/data5.csv', '(CCPP, 9568x5, gb, 5%):')
data_processing('/home/tyamana/missing_values/datasets/CCPP/data15.csv', '(CCPP, 9568x5, gb, 15%):')
data_processing('/home/tyamana/missing_values/datasets/CCPP/data30.csv', '(CCPP, 9568x5, gb, 30%):')
data_processing('/home/tyamana/missing_values/datasets/CCPP/data50.csv', '(CCPP, 9568x5, gb, 50%):')

data_processing('/home/tyamana/missing_values/datasets/concrete/data5.csv', '(concrete, 891x6, gb, 5%):')
data_processing('/home/tyamana/missing_values/datasets/concrete/data15.csv', '(concrete, 891x6, gb, 15%):')
data_processing('/home/tyamana/missing_values/datasets/concrete/data30.csv', '(concrete, 891x6, gb, 30%):')
data_processing('/home/tyamana/missing_values/datasets/concrete/data50.csv', '(concrete, 891x6, gb, 50%):')

data_processing('/home/tyamana/missing_values/datasets/crime/data15.csv', '(crime, 499x127, gb, 15%):')
data_processing('/home/tyamana/missing_values/datasets/crime/data30.csv', '(crime, 499x127, gb, 30%):')
data_processing('/home/tyamana/missing_values/datasets/crime/data50.csv', '(crime, 499x127, gb, 50%):')

"""
data_processing('/home/tyamana/data_from_kaggle/titanic/data5.csv', '(titanic, 891x6, svm, 5%):')
data_processing('/home/tyamana/data_from_kaggle/titanic/data15.csv', '(titanic, 891x6, svm, 15%):')
data_processing('/home/tyamana/data_from_kaggle/titanic/data30.csv', '(titanic, 891x6, svm, 30%):')
data_processing('/home/tyamana/data_from_kaggle/titanic/data50.csv', '(titanic, 891x6, svm, 50%):')

data_processing('/home/tyamana/missing_values/datasets/spambase/data5.csv', '(spambase, 1150x57, svm, 5%):')
data_processing('/home/tyamana/missing_values/datasets/spambase/data15.csv', '(spambase, 1150x57, svm, 15%):')
data_processing('/home/tyamana/missing_values/datasets/spambase/data30.csv', '(spambase, 1150x57, svm, 30%):')
data_processing('/home/tyamana/missing_values/datasets/spambase/data50.csv', '(spambase, 1150x57, svm, 50%):')

data_processing('/home/tyamana/missing_values/datasets/diabets/data15.csv', '(diabets, 768x8, svm, 15%):')
data_processing('/home/tyamana/missing_values/datasets/diabets/data30.csv', '(diabets, 768x8, svm, 30%):')
data_processing('/home/tyamana/missing_values/datasets/diabets/data50.csv', '(diabets, 768x8, svm, 50%):')

data_processing('/home/tyamana/missing_values/datasets/defaultCreditCard/data5.csv', '(default, 3600x23, svm, 5%):')
data_processing('/home/tyamana/missing_values/datasets/defaultCreditCard/data15.csv', '(default, 3600x23, svm, 15%):')

data_processing('/home/tyamana/missing_values/datasets/defaultCreditCard/data30.csv', '(default, 3600x23, svm, 30%):')
data_processing('/home/tyamana/missing_values/datasets/defaultCreditCard/data50.csv', '(default, 3600x23, svm, 50%):')


data_processing('/home/tyamana/missing_values/datasets/eeg/data5.csv', '(eeg, 2472x14, svm, 5%):')
data_processing('/home/tyamana/missing_values/datasets/eeg/data15.csv', '(eeg, 2472x14, svm, 15%):')
data_processing('/home/tyamana/missing_values/datasets/eeg/data30.csv', '(eeg, 2472x14, svm, 30%):')
data_processing('/home/tyamana/missing_values/datasets/eeg/data50.csv', '(eeg, 2472x14, svm, 50%):')

data_processing('/home/tyamana/missing_values/datasets/banknot/data5.csv', '(banknot, 686x4, svm, 5%):')
data_processing('/home/tyamana/missing_values/datasets/banknot/data15.csv', '(banknot, 686x4, svm, 15%):')
data_processing('/home/tyamana/missing_values/datasets/banknot/data30.csv', '(banknot, 686x4, svm, 30%):')
data_processing('/home/tyamana/missing_values/datasets/banknot/data50.csv', '(banknot, 686x4, svm, 50%):')

data_processing('/home/tyamana/missing_values/datasets/occ/data5.csv', '(occ, 4072x5, svm, 5%):')
data_processing('/home/tyamana/missing_values/datasets/occ/data15.csv', '(occ, 4072x5, svm, 15%):')
data_processing('/home/tyamana/missing_values/datasets/occ/data30.csv', '(occ, 4072x5, svm, 30%):')
data_processing('/home/tyamana/missing_values/datasets/occ/data50.csv', '(occ, 4072x5, svm, 50%):')

data_processing('/home/tyamana/missing_values/datasets/waves/data5.csv', '(waves, 2500x22, svm, 5%):')
data_processing('/home/tyamana/missing_values/datasets/waves/data15.csv', '(waves, 2500x22, svm, 15%):')
data_processing('/home/tyamana/missing_values/datasets/waves/data30.csv', '(waves, 2500x22, svm, 30%):')
data_processing('/home/tyamana/missing_values/datasets/waves/data50.csv', '(waves, 2500x22, svm, 50%):')


#logreg_classification(Imputer(strategy='mean').fit_transform(X), Y, results, 'mean', 0.1, 1.5, 0.1)
neighbors_classification(Imputer(strategy='mean').fit_transform(X_test), y_test, results, 'mean', 3, 30, 3)
neighbors_regression(Imputer(strategy='class_median').fit_transform(X, Y), Y, results, 'class_median', 10, 20, 3)
neighbors_regression(Imputer(strategy='class_mean').fit_transform(X, Y), Y, results, 'class_mean', 10, 20, 3)
neighbors_regression(Imputer(strategy='class_kmeans').fit_transform(X, Y, **{'n_clusters': 5,
                                                                  'max_iter': 300,
                                                                  'n_init': 100,
                                                                  'init': 'k-means++',
                                                                  'n_jobs': 2,
                                                                  'random_state': 282}).sort_index(), Y.sort_index(), results, 'class_kmeans', 10, 20, 3)
neighbors_regression(Imputer(strategy='class_xgboost').fit_transform(X, Y).sort_index(), Y.sort_index(), results, 'class_xgboost', 10, 20, 3)
neighbors_regression(Imputer(strategy='class_knn').fit_transform(X, Y).sort_index(), Y.sort_index(), results, 'class_knn', 10, 20, 3)
neighbors_regression(Imputer(strategy='class_svm').fit_transform(X, Y).sort_index(), Y.sort_index(), results, 'class_svm', 10, 20, 3)

ridge_regression(Imputer(strategy='knn').fit_transform(X), Y, results, 'knn', 0.01, 0.15, 0.01)
ridge_regression(Imputer(strategy='svm').fit_transform(X), Y, results, 'svm', 0.01, 0.15, 0.01)
ridge_regression(Imputer(strategy='xgboost').fit_transform(X), Y, results, 'xgboost', 0.01, 0.15, 0.01)

ridge_regression(Imputer(strategy='kmeans').fit_transform(X, Y, **{'n_clusters': 5,
                                                                'max_iter': 300,
                                                                'n_init': 100,
                                                                'init': 'k-means++',
                                                                'n_jobs': 2,
                                                                'random_state': 282}),
              Y, results, 'kmeans5', 0.01, 0.15, 0.01)
"""


