#The initially is in the form of EarlyStage train(postrain), EarlyStagetest(postest)[Stage1 and Stage2], LateStage train(negtrain), LateStage test(negtest)[Satge 3 and Stage 4]
import numpy as np
import pickle

#data is read as numpy arrays
pos_train_numpy_array = np.genfromtxt('pos_train_normal.txt',dtype='unicode',delimiter=',')
neg_train_numpy_array = np.genfromtxt('neg_train_normal.txt',dtype='unicode', delimiter=',',skip_header=1)
pos_test_numpy_array = np.genfromtxt('pos_test_normal.txt',dtype='unicode', delimiter=',',skip_header=1)
neg_test_numpy_array = np.genfromtxt('neg_test_normal.txt',dtype='unicode', delimiter=',',skip_header=1)

#setting the target column: postrain:EarlyStage=1 and negtraing:LateStage=0
postrain = np.ones((pos_train_numpy_array.shape[0],1), dtype=float)
negtrain = np.zeros((neg_train_numpy_array.shape[0],1), dtype=float)
postest = np.ones((pos_test_numpy_array.shape[0],1), dtype=float)
negtest = np.zeros((neg_test_numpy_array.shape[0],1), dtype=float)
targetcol_train= np.append(postrain, negtrain ,axis=0)
targetcol_test = np.append(postest, negtest ,axis=0)
targetcol = np.append(postrain, negtrain,axis=0)
targetcol = np.append(targetcol, postest, axis=0)
targetcol = np.append(targetcol, negtest, axis=0)


#combining the data as one
wholedataset = np.append(pos_train_numpy_array,neg_train_numpy_array, axis=0)
wholedataset = np.append(wholedataset,pos_test_numpy_array,axis=0)
wholedataset = np.append(wholedataset,neg_test_numpy_array,axis=0)
wholedataset = np.append(wholedataset,targetcol,axis=1)
print(wholedataset.shape)


with open('wholedataset.pkl', 'wb') as f:
    pickle.dump(wholedataset, f)
print('Done')


