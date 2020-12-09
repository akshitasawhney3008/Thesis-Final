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


def transform_to_genes(my_file, dict_of_words):
    list_of_string = []
    count = 0
    for line in my_file:
        count = count +1
        print("n_rs:",count)
        line_split = line.split('#')
        genes = line_split[0].rstrip(' ').split(' ')
        list_of_genes = []
        my_string = ''
        for gene in genes:
            if int(gene) in dict_of_words.values():
                indvalue = list(dict_of_words.values()).index(int(gene))
            list_of_genes.append(list(dict_of_words.keys())[indvalue])
        for g in list_of_genes:
            my_string = my_string + g + " "
        my_string = my_string + "#" + str(line_split[1].rstrip('\n'))
        list_of_string.append(my_string)
    mt_of_strings = np.asarray(list_of_string)
    return mt_of_strings

my_file1 = open("data_with_buckets_train_221f.csv", 'r')
file_read1 = my_file1.readlines()
dict_w1 = convert_to_dict(file_read1)
my_file3 = open("out_neg.txt",'r')
my_file_read3 = my_file3.readlines()
matrix_w1 = transform_to_genes(my_file_read3, dict_w1)
np.savetxt("negative_ism_to_genenames_221f.txt", matrix_w1, delimiter=" ", fmt="%s")