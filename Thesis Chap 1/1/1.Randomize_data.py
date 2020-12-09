#The data here is shuffled/randomized so that there no left bias
import numpy as np
import pickle
from sklearn.utils import shuffle

random_seed = 3
column_names = open('pos_train_normal.txt', 'r').readline().split(',') + ['Label']


with open("wholedataset.pkl", 'rb') as f:
    wholedataset = pickle.load(f)

#shuffle the data and save
X = wholedataset[1:, :-1]
Y = wholedataset[1:, -1]
wholedataset, target = shuffle(X, Y, random_state=random_seed)
wholedataset = np.append(wholedataset, target.reshape(-1,1), axis=1)

with open('wholedataset_shuffled.pkl', 'wb') as f:
    pickle.dump(wholedataset, f)
print('Done')


#add column names to the shuffled data
column_names_arr = np.asarray(column_names).reshape((1,len(column_names)))
wholedataset_col = np.append(column_names_arr,wholedataset,axis=0)

with open('wholedataset_shuffled_with_columnnames.pkl', 'wb') as f:
    pickle.dump(wholedataset_col, f)
print('Done')