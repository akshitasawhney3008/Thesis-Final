import numpy as np
import pickle
import random
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score

# Configuration section
iter = 5
cvCount = 8
seed = 42
wdiff = 0.30
wtest = 0.70
numSamples = 500

# Best parameter list
best_params = []

# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc

path = "C://Users//Arushi//PycharmProjects//ThesisChap2//fixedBuckets(10)//"

# Random search over parameters
for i in range(iter):
    X_train = np.load(path + 'final_train_binarydata_' + str(i) + '.npy')
    Y_train = np.load(path + 'final_train_labels_' + str(i) + '.npy')

    print(X_train.shape)
    booster = ['gbtree', 'gblinear', 'dart']
    eta = [x for x in np.linspace(0.1, 1, num=25)]
    gamma = [x for x in np.linspace(0, 100, num=75)]
    max_depth = [int(x) for x in np.linspace(1, 10, num=5)]
    tree_method = ['auto']
    grow_policy = ['depthwise', 'lossguide']

    grid = {'booster': booster,
            'eta': eta,
            'gamma': gamma,
            'max_depth': max_depth,
            'tree_method': tree_method,
            'grow_policy': grow_policy,
            }

    print('Searching')
    randomCombinations = random.sample(list(ParameterGrid(grid)), numSamples)
    score_list = []
    combination_list = []
    train_acc_list = []
    test_acc_list = []
    for combination in randomCombinations:
        skf = StratifiedKFold(n_splits=cvCount, random_state=seed, shuffle=True)
        s = 0
        tr_acc = 0
        te_acc = 0
        for train_idx, test_idx in skf.split(X_train, Y_train):
            split_x_train, split_x_test = X_train[train_idx], X_train[test_idx]
            y_true_train, y_true_test = Y_train[train_idx], Y_train[test_idx]
            xgb = XGBClassifier(**combination)
            clf = xgb.fit(split_x_train, y_true_train.ravel())
            y_pred_train = clf.predict(split_x_train)
            y_pred_test = clf.predict(split_x_test)
            score, fold_train_acc, fold_test_acc = myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test)
            s += score
            tr_acc += fold_train_acc
            te_acc += fold_test_acc
        combination_list.append(combination)
        score_list.append(s/cvCount)
        train_acc_list.append(tr_acc/cvCount)
        test_acc_list.append(te_acc/cvCount)

    req_idx = score_list.index(max(score_list))
    print(max(score_list))
    best_params.append(combination_list[req_idx])
    print(str(i) + '-' + str(train_acc_list[req_idx]) + '-' + str(test_acc_list[req_idx]))


with open('ListOfBestParamsRS.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print('Done')
