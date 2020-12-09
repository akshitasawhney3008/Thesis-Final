import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd


iter = 5
cvCount = 7
seed = 42


with open('ListOfBestParamsGS.pkl', 'rb') as f:
    best_params = pickle.load(f)


def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1 >= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)


path = 'Dataset/'
with open('dict_of_selected_genes_with_each_c(2).pkl', 'rb') as dict_file:
    dict_of_selectedgenes_with_c = pickle.load(dict_file)

train_acc = []
test_acc = []
keylist = []
for key, value in dict_of_selectedgenes_with_c.items():
    overall_accuracy_train = 0
    overall_accuracy_test = 0
    for i in range(iter):
        X_train = np.load(path + 'transformed_train_data_' + str(i) + '.npy')
        Y_train = np.load(path + 'transformed_train_labels_' + str(i) + '.npy')
        p = best_params[str(key) + '-' + str(i)]
        list_of_gene_numbers = list(map(int, value))

        X_train = X_train[:, list_of_gene_numbers]
        X_train = X_train.astype('float')
        X_train = normalize(X_train)
        Y_train = Y_train.astype('float')
        Y_train = Y_train.astype(int)

        skf = StratifiedKFold(n_splits=cvCount, random_state=seed)
        fold_accuracy_train = 0
        fold_accuracy_test = 0
        for train_index, test_index in skf.split(X_train, Y_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            Y_tr, Y_te = Y_train[train_index], Y_train[test_index]
            clf = SVC(C=p['C'], gamma=p['gamma'],probability=True).fit(X_tr, Y_tr.ravel())
            predictions_train = clf.predict(X_tr)
            predictions_test = clf.predict(X_te)
            accuracy_train = accuracy_score(Y_tr, predictions_train)
            accuracy_test = accuracy_score(Y_te, predictions_test)
            fold_accuracy_train += accuracy_train
            fold_accuracy_test += accuracy_test
        overall_accuracy_train += fold_accuracy_train / cvCount
        overall_accuracy_test += fold_accuracy_test / cvCount
    keylist.append(key)
    overall_accuracy_train = overall_accuracy_train / iter
    overall_accuracy_test = overall_accuracy_test / iter
    train_acc.append(overall_accuracy_train)
    test_acc.append(overall_accuracy_test)


diff_acc = []
for i in range(len(train_acc)):
    diff_acc.append(train_acc[i]-test_acc[i])
print(min(diff_acc))
req_idx = diff_acc.index(min(diff_acc))
print(req_idx)
print(test_acc[req_idx])
print(list(dict_of_selectedgenes_with_c.keys())[req_idx])
print(keylist[req_idx])