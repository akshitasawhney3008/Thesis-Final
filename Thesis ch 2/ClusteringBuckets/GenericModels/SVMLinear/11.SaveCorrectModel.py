import pickle
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np


iter = 5

with open('ListOfBestParamsRS.pkl', 'rb') as f:
    best_params = pickle.load(f)

path = "C://Users//Arushi//PycharmProjects//ThesisChap2//ClusteringBuckets//"


for i in range(iter):
    X_train = np.load(path +'final_train_binarydata_' + str(i) + '.npy')
    Y_train = np.load(path +'final_train_labels_' + str(i) + '.npy')
    bp = best_params[i]

    X_train = X_train.astype('float')
    X_train = normalize(X_train)
    Y_train = Y_train.astype('float')
    Y_train = Y_train.astype(int)

    clf = LinearSVC(C=bp['C'], max_iter=10000, tol=1e-4)
    clf_sigmoid = CalibratedClassifierCV(clf, cv=4, method='sigmoid').fit(X_train, Y_train.ravel())

    with open('Model_ism_linear' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(clf_sigmoid, f)
