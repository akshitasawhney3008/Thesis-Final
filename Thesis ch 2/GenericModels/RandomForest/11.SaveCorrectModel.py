import pickle
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import numpy as np


iter = 5

with open('ListOfBestParamsRS.pkl', 'rb') as f:
    best_params = pickle.load(f)

pathd = "C://Users//Arushi//PycharmProjects//ThesisChap2//Dataset//"

path = "C://Users//Arushi//PycharmProjects//ThesisChap2//fixedBuckets(10)//"
genenamesFile = open(path + "FinalFeaturesFile.txt",'r').readline().rstrip('\n').split(';')

list_of_top_genes_fin =[]

for i in range(iter):
    list_of_top_genes = []
    X_train = np.load(path +'final_train_binarydata_' + str(i) + '.npy')
    Y_train = np.load(path +'final_train_labels_' + str(i) + '.npy')
    bp = best_params[i]

    X_train = X_train.astype('float')
    X_train = normalize(X_train)
    Y_train = Y_train.astype('float')
    Y_train = Y_train.astype(int)

    clf = RandomForestClassifier(n_estimators=bp['n_estimators'], bootstrap=bp['bootstrap'], max_depth=bp['max_depth'],
                                 max_features=bp['max_features'], min_samples_leaf=bp['min_samples_leaf'],
                                 min_samples_split=bp['min_samples_split'],random_state=42).fit(X_train, Y_train.ravel())

    fi = (clf.feature_importances_)
    dict_of_fi = {}
    for i in range(X_train.shape[1]-1):
        dict_of_fi[genenamesFile[i]] = fi[i]

    list_of_fi = sorted(dict_of_fi.items(), key=lambda x: x[1], reverse=True)
    for i in range(25):
        list_of_top_genes.append(list_of_fi[i][0])
    list_of_top_genes_fin.append(list_of_top_genes)
print(list(set.intersection(*map(set, list_of_top_genes_fin))))

    # with open('Model_ism' + str(i) + '.pkl', 'wb') as f:
    #     pickle.dump(clf, f)
