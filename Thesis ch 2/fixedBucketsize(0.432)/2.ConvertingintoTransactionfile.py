import numpy as np


def convert_to_dict(file_name):
    dict_of_words = {}
    count = 0
    for line in file_name:
        line_split = line.split(" ")
        last_it = line_split.pop(len(line_split)-1)
        for word in line_split:
            if word not in dict_of_words.keys():
                count = count+1
                if count == 987:
                    print(word)
                dict_of_words[word] = count
    return dict_of_words


def transform_to_transaction(my_file, dict_of_words):
    count = 0
    rows = []
    for line in my_file:
        line_split = line.split(" ")
        row = []
        last_it = line_split.pop(len(line_split)-1)
        for word in line_split:
            if word in dict_of_words.keys():
                row.append(dict_of_words.get(word))
        row.append(last_it)
        rows.append(row)
    rows_matrix = np.asarray(rows)
    return rows_matrix


my_file = open("data_with_buckets_train_221f.csv", 'r')
file_read = my_file.readlines()
dict_w = convert_to_dict(file_read)
matrix_w = transform_to_transaction(file_read, dict_w)
positive_list = []
negative_list = []
for row in matrix_w:
    if row[-1] == "1\n":
        positive_list.append(row)
    else:
        negative_list.append(row)
positive_arr = np.asarray(positive_list)
negative_arr = np.asarray(negative_list)
np.savetxt("positive_transaction_data_221f.txt", positive_arr[:, :-1], delimiter=" ", fmt="%s")
np.savetxt("negative_transaction_data_221f.txt", negative_arr[:, :-1], delimiter=" ", fmt="%s")

my_file = open("data_with_buckets_train_221f.csv", 'r')
file_read = my_file.readlines()
positive_list = []
negative_list = []
for row in file_read:
    row = row.split(' ')
    if row[-1] == "1\n":
        positive_list.append(row)
    else:
        negative_list.append(row)
positive_arr = np.asarray(positive_list)
negative_arr = np.asarray(negative_list)
np.savetxt("positive_genenames_221f.txt", positive_arr[:,:-1], delimiter=" ", fmt="%s")
np.savetxt("negative_genenames_221f.txt", negative_arr[:,:-1], delimiter=" ", fmt="%s")