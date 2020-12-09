import pickle
import pandas as pd
import numpy as np

numberOfInstances = 5
# K = 0.5500000000000003
K = 55
path_d = "C://Users//Arushi//PycharmProjects//ThesisChap3//"

genenamesFile = open(path_d + "transformedColumnNames221.txt",'r').readline().rstrip('\n').split(',')

path ="C://Users//Arushi//PycharmProjects//ThesisChap3//Dataset//"

selectedFeaturesfile = open(path_d+'SelectedFeatures.csv').readlines()
flag = 0
list_of_gene_numbers = []
for line in selectedFeaturesfile:
        list_of_gene_names = line.rstrip('\n').split(',')
        if len(list_of_gene_names) == K:
            for gene in list_of_gene_names:
                list_of_gene_numbers.append(genenamesFile.index(gene))



for iterator in range(numberOfInstances):
    with open(path + 'transformed_train_data_' + str(iterator) + '.npy', 'rb') as f:
        train_data = np.load(f)
    with open(path + 'transformed_train_labels_' + str(iterator) + '.npy', 'rb') as f:
        Y_train = np.load(f)

    with open(path + 'transformed_test_data_' + str(iterator) + '.npy', 'rb') as f:
        test_data = np.load(f)

    with open(path + 'transformed_test_labels_' + str(iterator) + '.npy', 'rb') as f:
        Y_test = np.load(f)


    print(len(list_of_gene_numbers))

    X_train = train_data[:, list_of_gene_numbers]
    X_test = test_data[:, list_of_gene_numbers].astype("float")



    np.save(path_d + "Transformed Data//transformed_train_data_" + str(iterator) + ".npy", X_train)
    np.save(path_d + "Transformed Data//transformed_test_data_" + str(iterator) + ".npy", X_test)
    np.save(path_d + "Transformed Data//transformed_train_labels_" + str(iterator) + ".npy", Y_train)
    np.save(path_d + "Transformed Data//transformed_test_labels_" + str(iterator) + ".npy", Y_test)

    print(X_train.shape)
    X_train = pd.DataFrame(X_train)
    print(X_train.drop_duplicates().shape)

    print(X_test.shape)
    X_test = pd.DataFrame(X_test)
    print(X_test.drop_duplicates().shape)