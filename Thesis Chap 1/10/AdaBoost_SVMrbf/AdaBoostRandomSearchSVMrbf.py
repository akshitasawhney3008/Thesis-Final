import numpy as np
import pickle
import random
from sklearn.preprocessing import normalize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.svm import SVC

# Configuration section
iter = 5
cvCount = 7
seed = 42
wdiff = 0.50
wtest = 0.50
numSamples = 500

# List of best parameters
best_params = []

# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc

path = "C://Users//Arushi//PycharmProjects//Final_Thesis_chap1//9//"

# Random search over parameters
for i in range(iter):
    X_train = np.load(path +'transformed_train_data_' + str(i) + '.npy')
    Y_train = np.load(path +'transformed_train_labels_' + str(i) + '.npy')
    X_train = normalize(X_train)

    c = np.random.uniform(low=-5, high=12, size=(500,))
    c = np.power(2,c)
    c = c.tolist()
    base_estimator = []
    gamma = np.random.uniform(low=-15, high=3, size=(200,))
    gamma = np.power(2,gamma)
    gamma = gamma.tolist()
    grid = {'C': c, 'gamma': gamma}

    randomCombinationsSVM = random.sample(list(ParameterGrid(grid)), numSamples)

    for comb in randomCombinationsSVM:
        base_estimator.append(SVC(**comb, max_iter=2000,probability=True))

    n_estimators = [int(x) for x in np.linspace(25, 250, num=100)]
    algorithm = ['SAMME']

    grid = {'base_estimator': base_estimator,
            'n_estimators': n_estimators,
            'algorithm': algorithm
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
            adb = AdaBoostClassifier(**combination)
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
    print(str(i) + ',' + str(max(score_list)))
    req_idx = score_list.index(max(score_list))
    best_params.append(combination_list[req_idx])
    print(str(i) + '-' + str(train_acc_list[req_idx]) + '-' + str(test_acc_list[req_idx]))

with open('ListOfBestParamsRS.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print('Done')
