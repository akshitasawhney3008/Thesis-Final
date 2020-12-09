import numpy as np
import pickle
from scipy import stats
import math
import matplotlib.pyplot as plt


def split_data_pos_neg(data):
    total_pos = []
    total_neg = []
    flag = 0
    for row in data:
        if flag == 0:
            flag = 1
        else:
            if row[-1] == '1.0':
                total_pos.append(row)
            else:
                total_neg.append(row)

    total_neg = np.asarray(total_neg).astype('float')
    total_pos = np.asarray(total_pos).astype('float')
    return total_pos,total_neg


with open("wholedataset_shuffled_with_columnnames.pkl", 'rb') as f:
    completedata = pickle.load(f)

column_names = completedata[0, :]
list_of_gene_names = column_names.reshape(1,-1).tolist()[0]

Early_stage, Late_stage = split_data_pos_neg(completedata)
number_of_cols = Early_stage.shape[1]-1


fw = open('KS-Test_out.txt', 'w')
d_stat = {}
p_val = []

#making a dictionary of each gene with its ks_metric(d)
for i in range(number_of_cols):
    d, p = stats.ks_2samp(Early_stage[:, i], Late_stage[:, i])
    d_stat[i] = d
    p_val.append(p)
d_stat_list = sorted(d_stat.items(), key=lambda x: x[1])

#plot a graph 'Statistic measuring the largest distance between the EDF- Increasing Order'
x_list = []
d_stat_list1 = []
gene_number_list = []
for tup in d_stat_list:
    tup = list(tup)
    gene_number_list.append(tup[0])
    d_stat_list1.append(tup[1])
x_list = np.arange(0, number_of_cols, 1)
x_list = x_list.tolist()
plt.plot(x_list, d_stat_list1, 'red')

plt.ylabel('Distance between each gene in early and late respectively')
plt.title('Statistic measuring the largest distance between the EDF- Increasing Order')
plt.legend()
plt.show()

#Filtering the genes on the basis of ks metric
ks_genes = []
#we have kept d=0.12 so that we do not filter out many genes adnnd are left with sufficient to go ahead with other tests
for key,val in d_stat.items():
    if val > 0.12 and p_val[key] < 0.05:
        ks_genes.append(key)
        print(key)
        fw.write(str(key)+";"+list_of_gene_names[key] + '\n')

print(len(ks_genes))
