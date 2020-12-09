import numpy as np
import pandas as pd

iter = 5
path = 'Dataset/'


for i in range(iter):
    myTrainingData = np.load(path + 'transformed_train_data_' + str(i) + '.npy')
    myTrainingLabels= np.load(path + 'transformed_train_labels_' + str(i) + '.npy')

    CombinedData = np.append(myTrainingData,myTrainingLabels.reshape(-1,1), axis=1)

    PositiveLabelIndices = np.where(myTrainingLabels == 1)[0]
    NegativeLabelIndices = np.where(myTrainingLabels == 0)[0]

    if i == 0:
        PositiveData = CombinedData[PositiveLabelIndices,:-1]
        NegativeData = CombinedData[NegativeLabelIndices,:-1]
    else:
        PositiveData = np.append(PositiveData, CombinedData[PositiveLabelIndices,:-1],axis=0)
        NegativeData = np.append(NegativeData, CombinedData[NegativeLabelIndices,:-1],axis=0)

print(PositiveData.shape)
print(NegativeData.shape)

PositiveDataDF = pd.DataFrame(PositiveData).drop_duplicates()
PositiveDataDF = PositiveDataDF.transpose()
NegativeDataDF = pd.DataFrame(NegativeData).drop_duplicates()
NegativeDataDF = NegativeDataDF.transpose()

print(PositiveDataDF.shape)
print(NegativeDataDF.shape)

PositiveDataDF.to_csv("Pos_221.csv", sep=',',index=False, header=False)
NegativeDataDF.to_csv("Neg_221.csv", sep=',',index=False, header=False)

print(PositiveDataDF.shape)
print(NegativeDataDF.shape)







