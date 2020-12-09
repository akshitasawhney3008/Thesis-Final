import pickle
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
import numpy as np


iter = 5
seed = 42

with open('ListOfBestParamsRSXGB.pkl', 'rb') as f:
    best_params = pickle.load(f)



path = "C://Users//Arushi//PycharmProjects//ThesisChap3//Transformed Data//"


for i in range(iter):
    X_train = np.load(path + 'transformed_train_data_' + str(i) + '.npy')
    Y_train = np.load(path + 'transformed_train_labels_' + str(i) + '.npy')
    bp = best_params[i]

    X_train = X_train.astype('float')
    X_train = normalize(X_train)
    Y_train = Y_train.astype('float')
    Y_train = Y_train.astype(int)

    clf = XGBClassifier(booster=bp['booster'], eta=bp['eta'], gamma=bp['gamma'], max_depth=bp['max_depth'],
                        tree_method=bp['tree_method'], grow_policy=bp['grow_policy'], random_state=seed).fit(X_train,
                                                                                                             Y_train.ravel())

    with open('Model_xgb_nw' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
