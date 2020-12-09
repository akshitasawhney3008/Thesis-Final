import numpy as np
import pandas as pd
import csv
import pickle
import random
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed


class MyClass:
    def __init__(self, klist, paramlist):
        self.klist = klist
        self.paramlist = paramlist


# Configuration section
iter = 5
cvCount = 7
seed = 42
wdiff = 0.4
wtest = 0.6

GridSearchDict = dict()

def getList(val):
    tempList = []
    flag = 0
    charList = list(val)
    i=0
    while i < len(charList):
        if flag == 0:
            if charList[i] =='\'':
                flag = 1
        else:
            for j in range(i+1, len(charList)):
                if charList[j] == '\'':
                    tempList.append(int(val[i: j]))
                    i = j+1
                    flag = 0
                    break
        i += 1
    return tempList

# Load list of best parameters from Random Search
with open('new_dict_of_randomsearch_bestparams_rbf.pkl', 'rb') as f:
    random_params_dict = pickle.load(f)

# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc

def Stratified_kfold(X_train, Y_train, combination):
    skf = StratifiedKFold(n_splits=cvCount, random_state=seed)
    s = 0
    tr_acc = 0
    te_acc = 0
    for train_idx, test_idx in skf.split(X_train, Y_train):
        split_x_train, split_x_test = X_train[train_idx], X_train[test_idx]
        y_true_train, y_true_test = Y_train[train_idx], Y_train[test_idx]
        svm = SVC(**combination)
        clf = svm.fit(split_x_train, y_true_train.ravel())
        y_pred_train = clf.predict(split_x_train)
        y_pred_test = clf.predict(split_x_test)
        score, fold_train_acc, fold_test_acc = myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test)
        s += score
        tr_acc += fold_train_acc
        te_acc += fold_test_acc
    return combination, s/cvCount, tr_acc/cvCount, te_acc/cvCount


with open('dict_of_common_genes_with_k.csv') as csv_file:
    reader = csv.reader(csv_file)
    dict_of_common_genes_with_k = dict(reader)


# Grid search over parameters
for key, value in dict_of_common_genes_with_k.items():
    if float(key) >= 0.15 and float(key) <= 0.20:
        print(key)
        grid_k_list = []
        list_of_gene_numbers = getList(value)
        trainaccuracylist = []
        testaccuracylist = []

        for i in range(iter):
            train_data = pd.read_csv("new_train_data_" + str(i) + ".csv").values

            myobj = random_params_dict[str(key) + '-' + str(i)]
            p = myobj
            X_train = train_data[1:, list_of_gene_numbers]
            X_train = X_train.astype('float')
            X_train = normalize(X_train)
            Y_train = train_data[1:, -1]
            Y_train = Y_train.astype('float')
            Y_train = Y_train.astype(int)
            Y_train = Y_train.astype(int)

            val = p['C']
            gval = p['gamma']

            C = [x for x in np.linspace(max(0.1, val - 0.02), min(5, val + 0.02), num=20)]
            Gamma = [x for x in np.linspace(max(0.001, gval - 0.02), min(1, gval + 0.02), num=20)]

            param_grid = {
                'C': C,
                'gamma': Gamma
            }

            print('Searching')
            combinations = list(ParameterGrid(param_grid))

            print("parallel loop started")

            r = Parallel(n_jobs=-1, verbose=0)(delayed(Stratified_kfold)(X_train, Y_train, combination) for combination in combinations)

            comb, score, train_acc, test_acc = zip(*r)

            combination_list = list(comb)

            score_list = list(score)

            train_acc_list = list(train_acc)

            test_acc_list = list(test_acc)

            req_idx = score_list.index(max(score_list))
            print(req_idx)
            trainaccuracylist.append(train_acc_list[req_idx])
            testaccuracylist.append(test_acc_list[req_idx])
            GridSearchDict[str(key) + '-' + str(i)] = combination_list[req_idx]

        print('Train acc = ' + str(sum(trainaccuracylist) / iter))
        print('Test acc = ' + str(sum(testaccuracylist) / iter))

with open('ListOfBestParamsGS.pkl', 'wb') as f:
    pickle.dump(GridSearchDict, f)

print('Done')
