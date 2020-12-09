import numpy as np


def ranges(min_range,max_range, nb):
    step = (max_range-min_range) / nb
    list_of_range = []
    for i in range(nb):
        if i != nb-1:
            start_range = min_range+(step*i)
            end_range = min_range+(step*(i+1))
        else:
            start_range = min_range + (step * i)
            end_range = max_range

        list_of_range.append("{},{}".format(start_range,end_range))
    return list_of_range


all_list_of_ranges = []

column_file = open("transformedColumnNames221.txt", 'r')
column_file_read = column_file.readline()
column_names = column_file_read.rstrip('\n').split(',')
column_names = np.asarray(column_names)


column_names = column_names.reshape(1,-1)
gene_names_list = column_names.tolist()[0]
print('Making arrays/n')
#
pos_train_numpy_array_or = np.genfromtxt('Pos_221.csv',dtype='unicode',delimiter=',')
neg_train_numpy_array_or = np.genfromtxt('Neg_221.csv',dtype='unicode', delimiter=',')

pos_train_numpy_array = pos_train_numpy_array_or.transpose()
neg_train_numpy_array = neg_train_numpy_array_or.transpose()
print(pos_train_numpy_array.shape)
print(neg_train_numpy_array.shape)

postrain = np.ones((pos_train_numpy_array.shape[0],1), dtype=int)
negtrain = np.zeros((neg_train_numpy_array.shape[0],1), dtype=int)

targetcol_train= np.append(postrain, negtrain ,axis=0)
wholedataset = np.append(pos_train_numpy_array,neg_train_numpy_array, axis=0)



# wholedataset = wholedataset[:, :-1]
range_list_f = []
gene_col_discrete_list = []
for i in range(wholedataset.shape[1]):

    gene_col = wholedataset[:,i]
    gene_col = list(gene_col)
    gene_col = list(map(float, gene_col))
    max_gene_col = max(gene_col)
    min_gene_col = min(gene_col)
    list_of_ranges = ranges(float(min_gene_col),float(max_gene_col),10)
    all_list_of_ranges.append(list_of_ranges)
    range_list_f.append(list_of_ranges)
    gene_col_discrete = []
    for gene_exp in gene_col:
        for rnge in list_of_ranges:
            range_list = rnge.split(',')
            if float(gene_exp) >= float(range_list[0]) and float(gene_exp) <= float(range_list[1]):
                gene_col_discrete.append(str(str(list_of_ranges.index(rnge)) + gene_names_list[i]))
                break

        gene_col_discrete_list.append(gene_col_discrete)

print(len(all_list_of_ranges))
gene_col_discrete_arr = np.vstack(gene_col_discrete_list)
gene_col_discrete_arr = gene_col_discrete_arr.transpose()
gene_col_discrete_arr = np.append(gene_col_discrete_arr,targetcol_train,axis=1)

np.savetxt("data_with_buckets_train_221f.csv", gene_col_discrete_arr, delimiter=" ", fmt="%s")

AllRangesFile = open('AllRangesFile.txt', 'w')

for rng in all_list_of_ranges:
    AllRangesFile.write(';'.join(rng))
    AllRangesFile.write('\n')
