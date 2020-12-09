import numpy as np
import random
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from joblib import Parallel, delayed

iter = 5
cvCount = 8
seed = 42
wdiff = 0.3
wtest = 0.7
numSamples = 800

best_params = []

def Stratified_kfold(X_train, Y_train, combination):
    skf = StratifiedKFold(n_splits=cvCount, random_state=seed)
    s = 0
    tr_acc = 0
    te_acc = 0
    for train_idx, test_idx in skf.split(X_train, Y_train):
        split_x_train, split_x_test = X_train[train_idx], X_train[test_idx]
        y_true_train, y_true_test = Y_train[train_idx], Y_train[test_idx]
        rf = RandomForestClassifier(**combination)
        clf = rf.fit(split_x_train, y_true_train.ravel())
        y_pred_train = clf.predict(split_x_train)
        y_pred_test = clf.predict(split_x_test)
        score, fold_train_acc, fold_test_acc = myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test)
        s += score
        tr_acc += fold_train_acc
        te_acc += fold_test_acc
    return combination, s / cvCount, tr_acc / cvCount, te_acc / cvCount

# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc


path = "C://Users//Arushi//PycharmProjects//ThesisChap2//fixedBuckets(10)//"

for i in range(iter):
    X_train = np.load(path + 'final_train_binarydata_' + str(i) + '.npy')
    Y_train = np.load(path + 'final_train_labels_' + str(i) + '.npy')

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=300, num=150)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 100, num=50)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    print('Searching')
    randomCombinations = random.sample(list(ParameterGrid(grid)), numSamples)
    r = Parallel(n_jobs=4, verbose=10)(
        delayed(Stratified_kfold)(X_train, Y_train, combination) for combination in randomCombinations)
    combination, score, train_acc, test_acc = zip(*r)
    combination_list = (list(combination))

    score_list = list(score)
    train_acc_list = list(train_acc)
    test_acc_list = (list(test_acc))
    req_idx = score_list.index(max(score_list))
    print(max(score_list))
    best_params.append(combination_list[req_idx])
    print(str(i) + '-' + str(train_acc_list[req_idx]) + '-' + str(test_acc_list[req_idx]))

with open('ListOfBestParamsRS.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print('Done')