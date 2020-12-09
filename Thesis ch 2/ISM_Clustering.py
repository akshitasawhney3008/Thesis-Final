import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def ranges(distribution):
    score_list = []
    pred_list = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, n_jobs=2)
        km_pred = km.fit_predict(np.asarray(distribution).reshape(-1, 1))
        score = silhouette_score(np.asarray(distribution).reshape(-1, 1), km_pred, metric='euclidean')
        score_list.append(score)
        pred_list.append(km_pred)
    req_idx = score_list.index(max(score_list))
    preds = pred_list[req_idx]
    return preds

all_list_of_ranges = []

column_file = open("transformedColumnNames221.txt", 'r')
column_file_read = column_file.readline()
column_names = column_file_read.rstrip('\n').split(',')
column_names = np.asarray(column_names)


column_names = column_names.reshape(1,-1)
gene_names_list = column_names.tolist()[0]
print('Making arrays/n')
#
pos_train_numpy_array_or = np.genfromtxt('Po221.csv',dtype='unicode',delimiter=',')[1:,1:]
neg_train_numpy_array_or = np.genfromtxt('Neg221.csv',dtype='unicode', delimiter=',')[1:,1:]

pos_train_numpy_array = pos_train_numpy_array_or.transpose()
neg_train_numpy_array = neg_train_numpy_array_or.transpose()

print(pos_train_numpy_array.shape)
pos_train_numpy_array = pd.DataFrame(pos_train_numpy_array).drop_duplicates()
print(pos_train_numpy_array.shape)


postrain = np.ones((pos_train_numpy_array.shape[0],1), dtype=int)
negtrain = np.zeros((neg_train_numpy_array.shape[0],1), dtype=int)

targetcol_train= np.append(postrain, negtrain ,axis=0)
wholedataset = np.append(pos_train_numpy_array,neg_train_numpy_array, axis=0)
print(wholedataset.shape)
wholedataset = pd.DataFrame(wholedataset).drop_duplicates()
print(wholedataset.shape)


# wholedataset = wholedataset[:, :-1]
range_list_f = []
gene_col_discrete_list = []

for i in range(wholedataset.shape[1]):
    gene_col = wholedataset[:,i]
    gene_col = list(gene_col)
    gene_col = list(map(float, gene_col))
    bucketed_genes = ranges(gene_col)



    # Change code after this. the method call returns clustered gene expression values. modify accordingly


    #
    #
    #
    #
    # list_of_ranges = ''
    # all_list_of_ranges.append(list_of_ranges)
    # range_list_f.append(list_of_ranges)
    # gene_col_discrete = []
    # for gene_exp in gene_col:
    #     for rnge in list_of_ranges:
    #         range_list = rnge.split(',')
    #         if float(gene_exp) >= float(range_list[0]) and float(gene_exp) <= float(range_list[1]):
    #             gene_col_discrete.append(str(str(list_of_ranges.index(rnge)) + gene_names_list[i]))
    #             break
    #
    # gene_col_discrete_list.append(gene_col_discrete)

print(len(all_list_of_ranges))
gene_col_discrete_arr = np.vstack(gene_col_discrete_list)
gene_col_discrete_arr = gene_col_discrete_arr.transpose()
gene_col_discrete_arr = np.append(gene_col_discrete_arr,targetcol_train,axis=1)

np.savetxt("data_with_buckets_train_221f.csv", gene_col_discrete_arr, delimiter=" ", fmt="%s")

AllRangesFile = open('AllRangesFile.txt', 'w')

for rng in all_list_of_ranges:
    AllRangesFile.write(';'.join(rng))
    AllRangesFile.write('\n')
