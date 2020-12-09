import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_curve,auc, accuracy_score, matthews_corrcoef, f1_score
import pickle
from sklearn.preprocessing import normalize


iterations = 5
# threshold = 0.5

def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1>= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)

def getResults(predProb, labels):
    thresholdList = []
    precisionList = []
    recallList = []
    aucList = []
    accuracyListtr = []
    accuracyList = []
    mcList = []
    f1scoreList = []


    for threshold in thresholdRange:
        matrixPredictions = getPredictionsGivenThreshold(predProb, threshold)
        precision = precision_score(labels, matrixPredictions)
        recall = recall_score(labels, matrixPredictions)
        fpr, tpr, thresholds = roc_curve(labels, matrixPredictions, pos_label=1)
        auroc = auc(fpr, tpr)
        accuracy = accuracy_score(labels, matrixPredictions)
        matthewsCoeff = matthews_corrcoef(labels, matrixPredictions)
        f1score = f1_score(labels, matrixPredictions)

        thresholdList.append(threshold)
        precisionList.append(precision)
        recallList.append(recall)
        aucList.append(auroc)
        accuracyList.append(accuracy)
        mcList.append(matthewsCoeff)
        f1scoreList.append(f1score)
    print(max(accuracyList))
    ind = accuracyList.index((max(accuracyList)))

    print('Threshold: ' + str(thresholdList[ind]))
    print('Precision: ' + str(precisionList[ind]))
    print('Recall: ' + str(recallList[ind]))
    print('F1: ' + str(f1scoreList[ind]))
    print('Accuracy: ' + str(accuracyList[ind]))
    print('AUROC: ' + str(aucList[ind]))
    print('MCC: ' + str(mcList[ind]) + '\n')
    return max(accuracyList),precisionList[ind],recallList[ind], f1scoreList[ind],aucList[ind],mcList[ind]


path = "C://Users//Arushi//PycharmProjects//ThesisChap3//Transformed Data//"
listOfPredictionProbabilities = []
actualpredictions = []

# genenamesFile = open("transformedColumnNames221.txt",'r').readline().rstrip('\n').split(',')
# selectedFeaturesfile = open('SelectedFeatures.csv').readlines()
# flag = 0
#
# list_of_gene_numbers = []
# for line in selectedFeaturesfile:
#
#
#         list_of_gene_names = line.rstrip('\n').split(',')
#         if len(list_of_gene_names) == 55:
#             for gene in list_of_gene_names:
#                 list_of_gene_numbers.append(genenamesFile.index(gene))
#             flag = 1


finacclist = []
finpre = []
finrec =[]
finf1 =[]
finauc =[]
finmcc =[]
thresholdRange = np.linspace(start=0.40, stop=0.60, num=500)
for i in range(iterations):
    X_test = np.load(path + 'transformed_test_data_' + str(i) + '.npy')
    Y_test = np.load(path + 'transformed_test_labels_' + str(i) + '.npy')

    # X_test = X_test[:, list_of_gene_numbers]
    X_test = X_test.astype('float')
    X_test = normalize(X_test)
    Y_test = Y_test.astype('float')
    Y_test = Y_test.astype(int)

    with open('Model_lr_nw' + str(i) + '.pkl', 'rb') as f:
        model = pickle.load(f)

    predictionsProb_file = open("predictionsProb_lr_nw  " + str(i) + ".csv", 'w')
    predictionProbabilities = model.predict_proba(X_test)

    for prob in predictionProbabilities:
        for pr in prob:
            predictionsProb_file.write(str(pr) + ',')
        predictionsProb_file.write('\n')
    acc,pre,rec,f1,au,mcc = getResults(predictionProbabilities, Y_test)
    finacclist.append(acc)
    finpre.append(pre)
    finrec.append(rec)
    finf1.append(f1)
    finauc.append(au)
    finmcc.append(mcc)

print(sum(finacclist)/iterations)
print(sum(finpre)/iterations)
print(sum(finrec)/iterations)
print(sum(finf1)/iterations)
print(sum(finauc)/iterations)
print(sum(finmcc)/iterations)
print('Done')