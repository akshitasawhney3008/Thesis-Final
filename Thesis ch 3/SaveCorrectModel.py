import pickle
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import numpy as np


iter = 5

with open('ListOfBestParamsGS.pkl', 'rb') as f:
    best_params = pickle.load(f)


path = 'Dataset/'
genenamesFile = open("transformedColumnNames221.txt",'r').readline().rstrip('\n').split(',')


selectedFeaturesfile = open('SelectedFeatures.csv').readlines()
for line in selectedFeaturesfile:
    list_of_gene_numbers = []
    list_of_gene_names = line.rstrip('\n').split(',')
    for gene in list_of_gene_names:
        list_of_gene_numbers.append(genenamesFile.index(gene))
    if len(list_of_gene_numbers) == 55:
        for i in range(iter):
            X_train = np.load(path + 'transformed_train_data_' + str(i) + '.npy')
            Y_train = np.load(path + 'transformed_train_labels_' + str(i) + '.npy')
            p = best_params[str(len(line)) + '-' + str(i)]

            X_train = X_train[:, list_of_gene_numbers]
            X_train = X_train.astype('float')
            X_train = normalize(X_train)
            Y_train = Y_train.astype('float')
            Y_train = Y_train.astype(int)

            clf = SVC(C=p['C'], gamma=p['gamma'], probability=True).fit(X_train, Y_train.ravel())

            with open('Model_221' + str(i) + '.pkl', 'wb') as f:
                pickle.dump(clf, f)
