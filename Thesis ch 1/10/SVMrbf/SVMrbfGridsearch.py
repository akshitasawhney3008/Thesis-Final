import numpy as np
import pickle
import random
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score


# Configuration section
iter = 5
cvCount = 8
seed = 42
wdiff = 0.30
wtest = 0.70

RandomSearchDict = dict()


# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc

with open('ListOfBestParamsRS.pkl', 'rb') as f:
    random_params_dict = pickle.load(f)

best_params = []

path = "C://Users//Arushi//PycharmProjects//Final_Thesis_chap1//9//"

trainaccuracylist = []
testaccuracylist = []

for i in range(iter):
    X_train = np.load(path + 'transformed_train_data_' + str(i) + '.npy')
    Y_train = np.load(path + 'transformed_train_labels_' + str(i) + '.npy')

    X_train = X_train.astype('float')
    X_train = normalize(X_train)
    Y_train = Y_train.astype('float')
    Y_train = Y_train.astype(int)

    myobj = random_params_dict[i]
    p = myobj
    val = p['C']
    gval = p['gamma']

    C = [x for x in np.linspace(max(0.1, val - 0.03), min(15, val + 0.03), num=50)]
    Gamma = [x for x in np.linspace(max(0.001, gval - 0.03), min(1, gval + 0.03), num=50)]

    param_grid = {
        'C': C,
        'gamma': Gamma
    }

    combinations = list(ParameterGrid(param_grid))
    score_list = []
    combination_list = []
    train_acc_list = []
    test_acc_list = []
    for combination in combinations:
        skf = StratifiedKFold(n_splits=cvCount, random_state=seed, shuffle=True)
        s = 0
        tr_acc = 0
        te_acc = 0
        for train_idx, test_idx in skf.split(X_train, Y_train):
            split_x_train, split_x_test = X_train[train_idx], X_train[test_idx]
            y_true_train, y_true_test = Y_train[train_idx], Y_train[test_idx]
            adb = SVC(**combination)
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

with open('ListOfBestParamGS.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print('Done')