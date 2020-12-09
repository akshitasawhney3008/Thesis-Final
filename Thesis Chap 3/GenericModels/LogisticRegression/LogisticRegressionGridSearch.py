import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

# Configuration section
iter = 5
cvCount = 8
seed = 42
wdiff = 0.0
wtest = 1.0

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
path = "C://Users//Arushi//PycharmProjects//ThesisChap3//Transformed Data//"
for i in range(iter):
    X_train = np.load(path + 'transformed_train_data_' + str(i) + '.npy')
    Y_train = np.load(path + 'transformed_train_labels_' + str(i) + '.npy')
    X_train = normalize(X_train)
    rsbp = best_params[i]

    if rsbp['solver'] == 'saga' or rsbp['solver'] == 'liblinear':
        penalty = ['l1','l2']
    else:
        penalty = [rsbp['penalty']]

    val = rsbp['C']

    if val <=3 :
        # Create the parameter grid based on the results of random search
        c = np.arange(0.1, val + 5, 0.1)
    else:
        # Create the parameter grid based on the results of random search
        c = np.arange(val - 3, val + 5, 0.1)
    c = c.tolist()
    solver = [rsbp['solver']]
    max_iter = [rsbp['max_iter']]

    grid = {"penalty": penalty,
            "C": c,
            'solver': solver,
            'max_iter': max_iter
           }

    print('Searching')
    combinations = list(ParameterGrid(grid))
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
            mlp = LogisticRegression(**combination)
            clf = mlp.fit(split_x_train, y_true_train.ravel())
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
