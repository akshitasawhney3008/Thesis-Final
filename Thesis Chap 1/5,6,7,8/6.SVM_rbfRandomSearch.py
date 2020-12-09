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


# Configuration section
iter = 5
cvCount = 7
seed = 42
wdiff = 0.4
wtest = 0.6
numSamples = 1000

RandomSearchDict = dict()


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


# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc


with open('dict_of_common_genes_with_k.csv') as csv_file:
    reader = csv.reader(csv_file)
    dict_of_common_genes_with_k = dict(reader)

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
    combination_list.append(combination)
    score_list.append(s / cvCount)
    train_acc_list.append(tr_acc / cvCount)
    test_acc_list.append(te_acc / cvCount)
    return combination_list[0], score_list[0], train_acc_list[0], test_acc_list[0]

bestparamdict = dict()
for key, value in dict_of_common_genes_with_k.items():
    if float(key) >= 0.15 and float(key) <= 0.20:
        print('Searching for k = ' + str(key))
        list_of_gene_numbers = getList(value)
        trainaccuracylist = []
        testaccuracylist = []
        c = [x for x in np.linspace(0.1, 5, num=100)]
        gamma = [x for x in np.linspace(0.001, 1, num=100)]
        grid = {'C': c, 'gamma': gamma}
        for i in range(iter):
            train_data = pd.read_csv("new_train_data_" + str(i) + ".csv").values
            X_train = train_data[1:, list_of_gene_numbers]
            X_train = X_train.astype('float')
            X_train = normalize(X_train)
            Y_train = train_data[1:, -1]
            Y_train = Y_train.astype('float')
            Y_train = Y_train.astype(int)
            Y_train = Y_train.astype(int)

            randomCombinations = random.sample(list(ParameterGrid(grid)), numSamples)
            score_list = []
            combination_list = []
            train_acc_list = []
            test_acc_list = []

            print("parallel loop started")

            r = Parallel(n_jobs=-1,verbose=10)(delayed(Stratified_kfold)(X_train,Y_train,combination) for combination in randomCombinations)
            combination, score, train_acc, test_acc= zip(*r)

            combination_list = (list(combination))

            score_list = list(score)
            train_acc_list = list(train_acc)
            test_acc_list = (list(test_acc))

            req_idx = score_list.index(max(score_list))
            trainaccuracylist.append(train_acc_list[req_idx])
            testaccuracylist.append(test_acc_list[req_idx])
            bestparamdict[str(key) + '-' + str(i)] = combination_list[req_idx]

        print('Train acc = ' + str(sum(trainaccuracylist)/iter))
        print('Test acc = ' + str(sum(testaccuracylist) / iter))

with open('new_dict_of_randomsearch_bestparams_rbf.pkl', 'wb') as f:
    pickle.dump(bestparamdict, f)
print('Done')
