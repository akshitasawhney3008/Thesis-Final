import pickle
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
import numpy as np


iter = 5
seed = 42

with open('ListOfBestParamsRS.pkl', 'rb') as f:
    best_params = pickle.load(f)

path = "C://Users//Arushi//PycharmProjects//ThesisChap2//fixedBuckets(10)//"



for i in range(iter):
    X_train = np.load(path +'final_train_binarydata_' + str(i) + '.npy')
    Y_train = np.load(path +'final_train_labels_' + str(i) + '.npy')


    X_train = X_train.astype('float')
    X_train = normalize(X_train)
    Y_train = Y_train.astype('float')
    Y_train = Y_train.astype(int)

    bp = best_params[i]
    clf = XGBClassifier(booster=bp['booster'], eta=bp['eta'], gamma=bp['gamma'], max_depth=bp['max_depth'],
                        tree_method=bp['tree_method'], grow_policy=bp['grow_policy'], random_state=seed).fit(X_train,
                                                                                                             Y_train.ravel())
    with open('Model_ism_xgb' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
