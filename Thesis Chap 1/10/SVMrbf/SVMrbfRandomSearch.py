import numpy as np
import pickle
import random
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVR
from sklearn.preprocessing import normalize
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import r2_score


# Configuration section
iter = 1
cvCount = 8
seed = 42
wdiff = 0.30
wtest = 0.70
numSamples = 200

RandomSearchDict = dict()


# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = r2_score(y_true_train, y_pred_train)
    test_acc = r2_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc



best_params = []

path = "C://Users//Arushi//PycharmProjects//Final_Thesis_chap1//9/"

trainaccuracylist = []
testaccuracylist = []
c = [x for x in np.linspace(0.1, 15, num=100)]
gamma = [x for x in np.linspace(0.001, 1, num=100)]
grid = {'C': c, 'gamma': gamma}
for i in range(iter):
    with open("train_data_shuffled.pkl", 'rb') as f:
        traindata= pickle.load(f)
    X_train = traindata[:, :-1]
    Y_train = traindata[:,-1]

    # X_train = np.load(path + 'transformed_train_data_' + str(i) + '.npy')
    # Y_train = np.load(path + 'transformed_train_labels_' + str(i) + '.npy')


    # X_train = X_train.astype('float')
    X_train = normalize(X_train)
    Y_train = Y_train.astype('float')
    # Y_train = Y_train.astype(int)

    randomCombinations = random.sample(list(ParameterGrid(grid)), numSamples)
    score_list = []
    combination_list = []
    train_acc_list = []
    test_acc_list = []
    count = 0
    for combination in randomCombinations:
        count = count+1
        print(count)
        skf = KFold(n_splits=cvCount, random_state=seed, shuffle=True)
        s = 0
        tr_acc = 0
        te_acc = 0
        for train_idx, test_idx in skf.split(X_train, Y_train):
            split_x_train, split_x_test = X_train[train_idx], X_train[test_idx]
            y_true_train, y_true_test = Y_train[train_idx], Y_train[test_idx]
            adb = SVR(**combination)
            clf = adb.fit(split_x_train, y_true_train.ravel())
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
    req_idx = score_list.index(max(score_list))
    best_params.append(combination_list[req_idx])
    print(str(i) + '-' + str(train_acc_list[req_idx]) + '-' + str(test_acc_list[req_idx]))

with open('ListOfBestParamsRS.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print('Done')
