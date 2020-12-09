import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score

# Configuration section
iter = 5
cvCount = 6
seed = 42
wdiff = 0.5
wtest = 0.5

# Load list of best parameters from Random Search
with open('ListOfBestParamsRS.pkl', 'rb') as f:
    best_params = pickle.load(f)

# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc

best_params_gs = []
# Grid search over parameters
for i in range(iter):
    X_train = np.load('X_train_' + str(i) + '.npy')
    Y_train = np.load('Y_train_' + str(i) + '.npy')
    rsbp = best_params[i]

    booster = [rsbp['booster']]
    eta = [x for x in np.linspace(max(0.1, rsbp['eta'] - 0.2), min(1, rsbp['eta'] + 0.2), num=10)]
    gamma = [x for x in np.linspace(max(0, rsbp['gamma'] - 20), min(100, rsbp['gamma'] + 20), num=100)]
    max_depth = [int(x) for x in np.linspace(max(1, rsbp['max_depth'] - 3), min(10, rsbp['max_depth'] + 3), num=6)]
    tree_method = [rsbp['tree_method']]
    grow_policy = [rsbp['grow_policy']]

    param_grid = {
        'booster': booster,
        'eta': eta,
        'gamma': gamma,
        'max_depth': max_depth,
        'tree_method': tree_method,
        'grow_policy': grow_policy
    }


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
            xgb = XGBClassifier(**combination)
            clf = xgb.fit(split_x_train, y_true_train.ravel())
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
