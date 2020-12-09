import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, matthews_corrcoef
import pandas as pd

# Configuration section
iter = 5
cvCount = 6
seed = 42
thresholdRange = np.linspace(start=0.46, stop=0.54, num=50)

# Load list of best parameters from Random Search
with open('ListOfBestParamsGS.pkl', 'rb') as f:
    best_params = pickle.load(f)


def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1>= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)


thresholdList = []
precisionList = []
recallList = []
aucList = []
accuracyList = []
mcList = []
for threshold in thresholdRange:
    print(threshold)
    overallPrecision = 0
    overallRecall = 0
    overallAuauc = 0
    overallAccuracy = 0
    overallMc = 0
    for i in range(iter):
        X_train = np.load('X_train_' + str(i) + '.npy')
        Y_train = np.load('Y_train_' + str(i) + '.npy')
        X_test = np.load('X_test_' + str(i) + '.npy')
        Y_test = np.load('Y_test_' + str(i) + '.npy')
        bp = best_params[i]
        clf = AdaBoostClassifier(base_estimator=bp['base_estimator'], n_estimators=bp['n_estimators'],
                                 algorithm=bp['algorithm'], random_state=seed).fit(X_train, Y_train.ravel())
        predictionsProb = clf.predict_proba(X_test)
        predictions = getPredictionsGivenThreshold(predictionsProb, threshold)
        precision = precision_score(Y_test, predictions)
        recall = recall_score(Y_test, predictions)
        auroc = roc_auc_score(Y_test, predictionsProb[:, 1])
        accuracy = accuracy_score(Y_test, predictions)
        matthewsCoeff = matthews_corrcoef(Y_test, predictions)

        overallPrecision += precision
        overallRecall += recall
        overallAuauc += auroc
        overallAccuracy +=accuracy
        overallMc += matthewsCoeff
    thresholdList.append(threshold)
    precisionList.append(overallPrecision / iter)
    recallList.append(overallRecall / iter)
    aucList.append(overallAuauc / iter)
    accuracyList.append(overallAccuracy / iter)
    mcList.append(overallMc / iter)

df = pd.DataFrame()
df['Threshold'] = thresholdList
df['Precision'] = precisionList
df['Recall'] = recallList
df['AUROC'] = aucList
df['Accuracy'] = accuracyList
df['MC'] = mcList
df.to_csv('Thresholding.csv', index=False)
print('Done')