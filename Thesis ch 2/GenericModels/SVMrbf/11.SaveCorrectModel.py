import pickle
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import numpy as np


iter = 5

with open('ListOfBestParamGS.pkl', 'rb') as f:
    best_params = pickle.load(f)

pathd = "C://Users//Arushi//PycharmProjects//ThesisChap2//Dataset//"

path = "C://Users//Arushi//PycharmProjects//ThesisChap2//fixedBuckets(10)//"
genenamesFile = open(pathd + "transformedColumnNames221.txt",'r').readline().rstrip('\n').split(',')


for i in range(iter):
    X_train = np.load(path +'final_train_binarydata_' + str(i) + '.npy')
    Y_train = np.load(path +'final_train_labels_' + str(i) + '.npy')
    p = best_params[i]

    X_train = X_train.astype('float')
    X_train = normalize(X_train)
    Y_train = Y_train.astype('float')
    Y_train = Y_train.astype(int)

    clf = SVC(C=p['C'], gamma=p['gamma'], probability=True).fit(X_train, Y_train.ravel())

    with open('Model_ism' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
