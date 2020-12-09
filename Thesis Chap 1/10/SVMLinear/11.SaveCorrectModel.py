import pickle
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np


iter = 5
seed = 42

with open('ListOfBestParamsGS.pkl', 'rb') as f:
    best_params = pickle.load(f)



path = "C://Users//Arushi//PycharmProjects//Final_Thesis_chap1//9//"


for i in range(iter):
    X_train = np.load(path + 'transformed_train_data_' + str(i) + '.npy')
    Y_train = np.load(path + 'transformed_train_labels_' + str(i) + '.npy')

    X_train = X_train.astype('float')
    X_train = normalize(X_train)
    Y_train = Y_train.astype('float')
    Y_train = Y_train.astype(int)

    bp = best_params[i]
    clf = LinearSVC(C=bp['C'], max_iter=3000, tol=1e-2)
    clf_sigmoid = CalibratedClassifierCV(clf, cv=4, method='sigmoid').fit(X_train, Y_train.ravel())
    with open('Model_svmlinear' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(clf_sigmoid, f)
