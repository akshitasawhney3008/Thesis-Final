#In this we split the wholedatset in 5 sets of train and test data , where each train has a ratio of 54%-45%(positive-negative) class
import numpy as np
import pickle
import collections
from sklearn.model_selection import train_test_split


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

    total_neg = np.asarray(total_neg)
    total_pos = np.asarray(total_pos)
    return total_pos, total_neg


with open("wholedataset_shuffled_with_columnnames.pkl", 'rb') as f:
    wholedataset = pickle.load(f)


column_names = wholedataset[0, :]
column_names = column_names.reshape(1,-1)

#divide the wholdedataset into early stage and late stage
early_stage, late_stage = split_data_pos_neg(wholedataset)

#patients in early stage: 306
#patients in late stage : 217

for i in range(5):

    #picked 250 rows at random from early stage and append it with late stage: making the wholedata having 54%-45% ratio of positive and negative
    indices = np.random.randint(0, early_stage.shape[0], 250)
    new_array = early_stage[indices]
    whole_data = np.append(new_array, late_stage, axis=0)

    #split this new whole data randomly into train and test
    X_train, X_test, y_train, y_test = train_test_split(whole_data[:, :-1], whole_data[:, -1], stratify=whole_data[:, -1], test_size=0.3)
    print(collections.Counter(y_train))
    print(collections.Counter(y_test))
    X_train = np.append(X_train,y_train.reshape(-1, 1), axis=1)
    train_data = np.append(column_names, X_train, axis=0)
    tr_es, tr_ls = split_data_pos_neg(train_data)
    print(tr_es.shape)
    print(tr_ls.shape)

    X_test = np.append(X_test, y_test.reshape(-1, 1),axis=1)
    test_data = np.append(column_names, X_test, axis=0)
    te_es, te_ls = split_data_pos_neg(test_data)
    print(te_es.shape)
    print(te_ls.shape)

    with open('new_train_data_' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    print('Done')

    with open('new_test_data_' + str(i) + '.pkl', 'wb') as f1:
        pickle.dump(test_data, f1)

    print('Done')