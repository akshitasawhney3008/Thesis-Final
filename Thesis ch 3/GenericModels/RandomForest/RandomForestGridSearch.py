import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score
import pickle

iter = 5
cvCount = 8
seed = 42
wdiff = 0.50
wtest = 0.50

with open('ListOfBestParamsRS.pkl', 'rb') as f:
    best_params = pickle.load(f)

# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc

best_params_gs = []

path = "C://Users//Arushi//PycharmProjects//ThesisChap2//"

for i in range(iter):
    X_train = np.load(path + 'final_train_binarydata_' + str(i) + '.npy')
    Y_train = np.load(path + 'final_train_labels_' + str(i) + '.npy')
    rp = best_params[i]

    if rp['min_samples_leaf'] == 1:
        # Create the parameter grid based on the results of random search
        param_grid = {
            'n_estimators': [int(x) for x in np.linspace(start=rp["n_estimators"]-10, stop=rp["n_estimators"]+10, num=20)],
            'min_samples_split': [rp['min_samples_split']-1, rp['min_samples_split'], rp['min_samples_split']+1],
            'min_samples_leaf': [rp['min_samples_leaf'], rp['min_samples_leaf']+1],
            'max_features': [rp['max_features']],
            'max_depth': [int(x) for x in np.linspace(start=max(1,rp['max_depth']-5), stop=rp['max_depth']+5, num=10)],
            'bootstrap': [rp['bootstrap']]
        }
    else:
        # Create the parameter grid based on the results of random search
        param_grid = {
            'n_estimators': [int(x) for x in np.linspace(start=rp['n_estimators']-10, stop=rp['n_estimators']+10, num=20)],
            'min_samples_split': [rp['min_samples_split']-1, rp['min_samples_split'], rp['min_samples_split']+1],
            'min_samples_leaf': [rp['min_samples_leaf'] - 1, rp['min_samples_leaf'], rp['min_samples_leaf']+1],
            'max_features': [rp['max_features']],
            'max_depth': [int(x) for x in np.linspace(start=max(1,rp['max_depth']-5), stop=rp['max_depth']+5, num=10)],
            'bootstrap': [rp['bootstrap']]
        }

    # Create a based model
    print('Searching')
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
            adb = RandomForestClassifier(**combination)
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
    best_params_gs.append(combination_list[req_idx])
    print(str(train_acc_list[req_idx]) + '-' + str(test_acc_list[req_idx]))

with open('ListOfBestParamsGS.pkl', 'wb') as f:
    pickle.dump(best_params_gs, f)

print('Done')
