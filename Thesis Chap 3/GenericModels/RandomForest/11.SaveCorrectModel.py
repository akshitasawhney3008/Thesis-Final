import pickle
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import numpy as np


iter = 5

with open('ListOfBestParamsRSRF.pkl', 'rb') as f:
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

    clf = RandomForestClassifier(n_estimators=bp['n_estimators'], bootstrap=bp['bootstrap'], max_depth=bp['max_depth'],
                                 max_features=bp['max_features'], min_samples_leaf=bp['min_samples_leaf'],
                                 min_samples_split=bp['min_samples_split']).fit(X_train, Y_train.ravel())

    with open('Model_rf_nw' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
