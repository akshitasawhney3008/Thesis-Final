import pickle
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import numpy as np


iter = 5
seed = 42

with open('ListOfBestParamsRS.pkl', 'rb') as f:
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

    clf = LogisticRegression(penalty=bp['penalty'], C=bp['C'],
                             solver=bp['solver'], max_iter=bp['max_iter'],
                             random_state=seed).fit(X_train, Y_train.ravel())

    with open('Model_lr_nw' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
