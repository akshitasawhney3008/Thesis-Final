import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, matthews_corrcoef
import pandas as pd

# Configuration section
iter = 5
cvCount = 7
seed = 42
thresholdRange = np.linspace(start=0.40, stop=0.60, num=500)
#thresholdRange = [0.578757515]
# Load list of best parameters from Random Search
with open('ListOfBestParamsRS.pkl', 'rb') as f:
    best_params = pickle.load(f)


def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1 >= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)

path = "C://Users//Arushi//PycharmProjects//ThesisChap2//fixedBuckets(10)//"

thresholdList = []
precisionList = []
recallList = []
aucList = []
accuracyList = []
for threshold in thresholdRange:
    print(threshold)
    overallPrecision = 0
    overallRecall = 0
    overallAuauc = 0
    overallAccuracy = 0
    overallMc = 0

    for i in range(iter):
        X_train = np.load(path + 'final_train_binarydata_' + str(i) + '.npy').astype(float)
        Y_train = np.load(path + 'final_train_labels_' + str(i) + '.npy').astype(float).astype(int)
        X_test = np.load(path+ 'final_test_binarydata_' + str(i) + '.npy').astype(float)
        Y_test = np.load(path+'final_test_labels_' + str(i) + '.npy').astype(float).astype(int)
        bp = best_params[i]

        clf = RandomForestClassifier(n_estimators=bp['n_estimators'],bootstrap=bp['bootstrap'], max_depth=bp['max_depth'], max_features=bp['max_features'], min_samples_leaf=bp['min_samples_leaf'], min_samples_split=bp['min_samples_split']).fit(X_train, Y_train.ravel())

        predictionsProb = clf.predict_proba(X_test)
        np.savetxt('pp_rf' + str(i) + '.csv', predictionsProb, delimiter=',')

        predictions = getPredictionsGivenThreshold(predictionsProb, threshold)
        precision = precision_score(Y_test, predictions)
        recall = recall_score(Y_test, predictions)
        auroc = roc_auc_score(Y_test, predictionsProb[:, 1])
        accuracy = accuracy_score(Y_test, predictions)

        overallPrecision += precision
        overallRecall += recall
        overallAuauc += auroc
        overallAccuracy +=accuracy
    thresholdList.append(threshold)
    precisionList.append(overallPrecision / iter)
    recallList.append(overallRecall / iter)
    aucList.append(overallAuauc / iter)
    accuracyList.append(overallAccuracy / iter)

df = pd.DataFrame()
df['Threshold'] = thresholdList
df['Precision'] = precisionList
df['Recall'] = recallList
df['AUROC'] = aucList
df['Accuracy'] = accuracyList
df.to_csv('Thresholding.csv', index=False)
print('Done')
