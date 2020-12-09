import pickle
import pandas as pd
import csv

numIter = 5

with open('dict_of_common_genes_with_k.pkl', 'rb') as f:
    mydict = pickle.load(f)

with open('dict_of_common_genes_with_k.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in mydict.items():
       writer.writerow([key, value])

for iterator in range(numIter):
    with open('new_train_data_' + str(iterator) + '.pkl', 'rb') as f:
        train = pickle.load(f)
    with open('new_test_data_' + str(iterator) + '.pkl', 'rb') as f:
        test = pickle.load(f)
    trainDf = pd.DataFrame(train)
    testDf = pd.DataFrame(test)
    trainDf.to_csv('new_train_data_' + str(iterator) + '.csv', index=False)
    testDf.to_csv('new_train_data_' + str(iterator) + '.csv', index=False)
print('Done')
