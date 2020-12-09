import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd
import csv

# Configuration section
iter = 5
cvCount = 7
seed = 42
thresholdRange = np.linspace(start=0.46, stop=0.54, num=50)

# Load list of best parameters from Random Search
with open('ListOfBestParamsGS.pkl', 'rb') as f:
    best_params= pickle.load(f)


def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1>= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)



with open('dict_of_common_genes_with_k.pkl', 'rb') as csv_file:
    dict_of_common_genes_with_k = pickle.load(csv_file)

for threshold in thresholdRange:
    keylist=[]
    thresholdList = []
    precisionList = []
    recallList = []
    aucList = []
    accuracyList = []
    mcList = []

    for key, value in dict_of_common_genes_with_k.items():
        if float(key)>0.14 and float(key)<0.58:
            print(key)
            print(threshold)
            overallPrecision = 0
            overallRecall = 0
            overallAuauc = 0
            overallAccuracy = 0
            overallMc = 0

            for i in range(iter):
                print(i)
                train_data = pd.read_csv("new_train_data_" + str(i) + ".csv").values

                myobj = best_params[str(key) + '-' + str(i)]
                p = myobj

                # list_of_gene_numbers = getList(value)
                list_of_gene_numbers = list(map(int, value))

                X_train = train_data[1:, list_of_gene_numbers]
                X_train = X_train.astype('float')
                X_train = normalize(X_train)
                Y_train = train_data[1:, -1]
                Y_train = Y_train.astype('float')
                Y_train = Y_train.astype(int)
                Y_train = Y_train.astype(int)

                skf = StratifiedKFold(n_splits=cvCount, random_state=seed)
                foldPrecision = 0
                foldRecall = 0
                foldAuauc = 0
                foldAccuracy = 0
                foldMc = 0
                for train_index, test_index in skf.split(X_train, Y_train):
                    X_tr, X_te = X_train[train_index], X_train[test_index]
                    Y_tr, Y_te = Y_train[train_index], Y_train[test_index]

                    clf = SVC(C=p['C'], gamma=p['gamma'],probability=True).fit(X_train, Y_train.ravel())
                    predictionsProb = clf.predict_proba(X_te)
                    predictions = getPredictionsGivenThreshold(predictionsProb, threshold)
                    precision = precision_score(Y_te, predictions)
                    recall = recall_score(Y_te, predictions)
                    fpr, tpr, thresholds = roc_curve(Y_te, predictions, pos_label=1)
                    auroc = auc(fpr, tpr)
                    accuracy = accuracy_score(Y_te, predictions)
                    matthewsCoeff = matthews_corrcoef(Y_te, predictions)

                    foldPrecision += precision
                    foldRecall += recall
                    foldAuauc += auroc
                    foldAccuracy += accuracy
                    foldMc += matthewsCoeff
                overallPrecision = overallPrecision + (foldPrecision/cvCount)
                overallRecall = overallRecall + (foldRecall/cvCount)
                overallAuauc = overallAuauc + (foldAuauc/cvCount)
                overallAccuracy = overallAccuracy + (foldAccuracy/cvCount)
                overallMc = overallMc + (foldMc/cvCount)

            keylist.append(key)
            thresholdList.append(threshold)
            precisionList.append(overallPrecision/iter)
            recallList.append(overallRecall/iter)
            aucList.append(overallAuauc/iter)

            accuracyList.append(overallAccuracy/iter)
            print(overallAccuracy / iter)
            mcList.append(overallMc/iter)

    df = pd.DataFrame()
    df["key"] = keylist
    df['Threshold'] = thresholdList
    df['Precision'] = precisionList
    df['Recall'] = recallList
    df['AUROC'] = aucList
    df['Accuracy'] = accuracyList
    df['MC'] = mcList
    df.to_csv('Thresholding.csv', index=False)
print('Done')
