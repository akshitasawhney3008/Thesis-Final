import numpy as np
import pandas as pd


iter = 5

def convertGEvaluestobinary(wholedataset,targetcol_train, listOfRangeOfallGenes, gene_names_list):
    gene_col_discrete_list = []
    for i in range(wholedataset.shape[1]):
        print(wholedataset.shape[1])
        gene_col = wholedataset[:, i]
        gene_col = list(gene_col)
        gene_col = list(map(float, gene_col))
        print(i)
        list_of_ranges = listOfRangeOfallGenes[i]
        gene_col_discrete = []
        for gene_exp in gene_col:
            flag = 0
            for rnge in list_of_ranges:
                range_list = rnge.split(',')
                if float(gene_exp) >= float(range_list[0]) and float(gene_exp) <= float(range_list[1]):
                    gene_col_discrete.append(str(str(list_of_ranges.index(rnge)) + gene_names_list[i]))
                    flag = 1
                    break
            if flag == 0:
                gene_col_discrete.append(str(len(list_of_ranges)) + gene_names_list[i])
        gene_col_discrete_list.append(gene_col_discrete)

    targetcol_train = targetcol_train.reshape((targetcol_train.shape[0], 1))
    gene_col_discrete_arr = np.vstack(gene_col_discrete_list)
    gene_col_discrete_arr = gene_col_discrete_arr.transpose()
    gene_col_discrete_arr = np.append(gene_col_discrete_arr, targetcol_train, axis=1)
    return gene_col_discrete_arr


def FinaliseDataforFeatures(genenames_with_buckets_arr, features):
    target =[]
    input_data_list_test =[]
    flag = 0
    for train_data in genenames_with_buckets_arr:
        input_data = []
        for f in features:
            list_f = f.split(',')
            count = 0
            for f in list_f:
                if f in train_data:
                    count = count + 1
            if count == len(list_f):
                input_data.append('1')
                flag = 1
            else:
                input_data.append('0')
        if flag == 1:
            input_data.append('0')
        else:
            input_data.append('1')
        target.append(float(train_data[-1].rstrip('\n')))
        input_data_list_test.append(input_data)

    return np.asarray(input_data_list_test).astype(float), np.asarray(target)


column_file = open("transformedColumnNames221.txt", 'r')
column_file_read = column_file.readline()
column_names = column_file_read.rstrip('\n').split(',')
column_names = np.asarray(column_names)
column_names = column_names.reshape(1,-1)
gene_names_list = column_names.tolist()[0]
path = 'C:/Users/Arushi/PycharmProjects/ThesisChap2/Dataset/'
ff = open("FinalFeaturesFile.txt", 'r').readline().rstrip('\n').split(';')
AllRangesFile = open('AllRangesFile.txt', 'r').readlines()
listOfRangeOfallGenes =[]
for listOfRange in AllRangesFile:
    listOfRangeOfallGenes.append(listOfRange.rstrip('\n').split(';'))


for i in range(iter):
    X_train = np.load(path +'transformed_train_data_' + str(i) + '.npy')
    Y_train = np.load(path +'transformed_train_labels_' + str(i) + '.npy')
    X_test = np.load(path + 'transformed_test_data_' + str(i) + '.npy')
    Y_test = np.load(path+'transformed_test_labels_' + str(i) + '.npy')

    gene_col_discrete_arr_train = convertGEvaluestobinary(X_train, Y_train, listOfRangeOfallGenes, gene_names_list)
    gene_col_discrete_arr_test = convertGEvaluestobinary(X_test,Y_test,listOfRangeOfallGenes, gene_names_list)

    final_train_arr , target_train = FinaliseDataforFeatures(gene_col_discrete_arr_train,ff)
    final_test_arr , target_test = FinaliseDataforFeatures(gene_col_discrete_arr_test,ff)

    np.save("final_train_binarydata_" + str(i) + ".npy", final_train_arr)
    np.save("final_test_binarydata_" + str(i) + ".npy", final_test_arr)
    np.save("final_train_labels_" + str(i) + ".npy", target_train)
    np.save("final_test_labels_" + str(i) + ".npy", target_test)
