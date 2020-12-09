import pickle
import pandas as pd
import numpy as np

numberOfInstances = 5
K = 0.5700000000000003

with open('dict_of_common_genes_with_k.pkl', 'rb') as f:
    dict_of_common_genes_with_k = pickle.load(f)

list_of_indices = list(map(int, dict_of_common_genes_with_k[K]))

file_read = open('new_train_data_0.csv', 'r').readline().split(',')
column_names_arr = np.asarray(file_read).reshape((1,len(file_read)))
column_names_arr = column_names_arr[:,list_of_indices]


for iterator in range(numberOfInstances):
    train_data = pd.read_csv("new_train_data_" + str(iterator) + ".csv").values
    test_data = pd.read_csv("new_test_data_" + str(iterator) + ".csv").values

    X_train = train_data[:, :-1]
    X_test = test_data[:, :-1]
    Y_train = train_data[:, -1].astype("int")
    Y_test = test_data[:, -1].astype("int")

    np.save("transformed_train_data_" + str(iterator) + ".npy", X_train)
    np.save("transformed_test_data_" + str(iterator) + ".npy", X_test)
    np.save("transformed_train_labels_" + str(iterator) + ".npy", Y_train)
    np.save("transformed_test_labels_" + str(iterator) + ".npy", Y_test)
