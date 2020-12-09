import numpy as np
import pickle


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


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    for i in range(len(p)):
        if p[i] == 0:
            p[i] = 0.0001
    for i in range(len(q)):
        if q[i] == 0:
            q[i] = 0.0001
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


with open("wholedataset_shuffled_with_columnnames.pkl", 'rb') as f:
    completedata = pickle.load(f)


filew = open("extracted_features_modified.txt", 'w')

total_pos, total_neg = split_data_pos_neg(completedata)

# print(len(total_neg))
# print(len(total_pos))

#read the file with genes filtered from ks test
f = open('KS-Test_out.txt', 'r')
lines = f.readlines()
list_of_gene_names_ks =[]
for el in lines:
    list_of_gene_names_ks.append(el.rstrip('\n'))

#from k = 0.05 to k= 0.60 we filter genes by computing kl divergence of each gene(early and late) and store common genes we get from ks test and kl divergence
final_list = []
k = 0.05

final_gene_count_list = []
final_train_acc_list = []
final_test_acc_list = []

#We have to convert each column of gene into a probability distribution of early an late and compute kl diverence between them
dict_of_common_genes_with_each_k = {}
while k <= 0.60:
    print(k)
    gene_name_list1 = []
    for i in range(0, total_pos.shape[1]-1):
        column_vector_pos = total_pos[:, i]
        column_vector_neg = total_neg[:, i]
        # Positive
        bp1 = 0
        bp2 = 0
        bp3 = 0
        bp4 = 0
        bp5 = 0
        bp6 = 0
        bp7 = 0
        bp8 = 0
        bp9 = 0
        bp10 = 0
        tot_pos_patients = total_pos.shape[0]
        min_range_p = float(min(total_pos[:, i]))
        max_range_p = float(max(total_pos[:, i]))
        min_range_n = float(min(total_neg[:, i]))
        max_range_n = float(max(total_neg[:, i]))
        if min_range_p < min_range_n:
            min_range = min_range_p
        else:
            min_range = min_range_n

        if max_range_p > max_range_n:
            max_range = max_range_p
        else:
            max_range = max_range_n

        diff = max_range - min_range
        step_size = (max_range - min_range)/10
        for j in range(0, total_pos.shape[0]):
            elt = float(total_pos[j, i])
            if min_range <= elt < min_range + step_size:
                bp1 += 1
            elif min_range + step_size <= elt < min_range +(2*step_size):
                bp2 += 1
            elif min_range +(2*step_size) <= elt < min_range +(3*step_size):
                bp3 += 1
            elif min_range +(3*step_size) <= elt < min_range +(4*step_size):
                bp4 += 1
            elif min_range +(4*step_size) <= elt < min_range +(5*step_size):
                bp5 += 1
            elif min_range +(5*step_size)<= elt < min_range +(6*step_size):
                bp6 += 1
            elif min_range +(6*step_size) <= elt < min_range +(7*step_size):
                bp7 += 1
            elif min_range +(7*step_size) <= elt < min_range +(8*step_size):
                bp8 += 1
            elif min_range +(8*step_size)<= elt < min_range +(9*step_size):
                bp9 += 1
            elif min_range +(9*step_size) <= elt <= max_range:
                bp10 += 1
        bp1 /= tot_pos_patients
        bp2 /= tot_pos_patients
        bp3 /= tot_pos_patients
        bp4 /= tot_pos_patients
        bp5 /= tot_pos_patients
        bp6 /= tot_pos_patients
        bp7 /= tot_pos_patients
        bp8 /= tot_pos_patients
        bp9 /= tot_pos_patients
        bp10 /= tot_pos_patients

        # Negative
        bn1 = 0
        bn2 = 0
        bn3 = 0
        bn4 = 0
        bn5 = 0
        bn6 = 0
        bn7 = 0
        bn8 = 0
        bn9 = 0
        bn10 = 0
        tot_neg_patients = total_neg.shape[0]
        for j in range(0, total_neg.shape[0]):
            elt = float(total_neg[j, i])
            if min_range <= elt < min_range + step_size:
                bn1 += 1
            elif min_range + step_size <= elt < min_range +(2*step_size):
                bn2 += 1
            elif min_range +(2*step_size) <= elt < min_range +(3*step_size):
                bn3 += 1
            elif min_range +(3*step_size) <= elt < min_range +(4*step_size):
                bn4 += 1
            elif min_range +(4*step_size) <= elt < min_range +(5*step_size):
                bn5 += 1
            elif min_range +(5*step_size)<= elt < min_range +(6*step_size):
                bn6 += 1
            elif min_range +(6*step_size) <= elt < min_range +(7*step_size):
                bn7 += 1
            elif min_range +(7*step_size) <= elt < min_range +(8*step_size):
                bn8 += 1
            elif min_range +(8*step_size)<= elt < min_range +(9*step_size):
                bn9 += 1
            elif min_range +(9*step_size) <= elt <= max_range:
                bn10 += 1

        bn1 /= tot_neg_patients
        bn2 /= tot_neg_patients
        bn3 /= tot_neg_patients
        bn4 /= tot_neg_patients
        bn5 /= tot_neg_patients
        bn6 /= tot_neg_patients
        bn7 /= tot_neg_patients
        bn8 /= tot_neg_patients
        bn9 /= tot_neg_patients
        bn10 /= tot_neg_patients

        total_p = bp1 + bp2 + bp3 + bp4 + bp5 + bp6 + bp7 + bp8 + bp9 + bp10
        total_n = bn1 + bn2 + bn3 + bn4 + bn5 + bn6 + bn7 + bn8 + bn9 + bn10

        list_p = [bp1, bp2, bp3, bp4, bp5, bp6, bp7, bp8, bp9, bp10]
        list_n = [bn1, bn2, bn3, bn4, bn5, bn6, bn7, bn8, bn9, bn10]

        final_list.append(kl_divergence(list_p, list_n))
        kld = kl_divergence(list_p, list_n)

        if kld >= k:
            gene_name_list1.append(str(i) + ";" + list_of_gene_names_ks[i])
    k = k + 0.01
    gene_name_set1 = set(gene_name_list1)

    common_genes = gene_name_set1.intersection(list(set(list_of_gene_names_ks)))
    common_genes_list = []

    filew.write("K= " + str(k) + '\n')
    for elt in list(common_genes):
        filew.write(str(elt) + '\n')
        common_genes_list.append(elt.split(';')[0])
    print(len(common_genes_list))
    dict_of_common_genes_with_each_k[k] = common_genes_list

with open('dict_of_common_genes_with_k.pkl', 'wb') as f:
    pickle.dump(dict_of_common_genes_with_each_k, f)
print('Done')