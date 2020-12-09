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
    # print('Precision: ' + str(precisionList[ind]))
    # print('Recall: ' + str(recallList[ind]))
    # print('F1: ' + str(f1scoreList[ind]))
    print('Accuracy: ' + str(accuracyList[ind]) + '\n')
    # print('AUROC: ' + str(aucList[ind]))
    # print('MCC: ' + str(mcList[ind]))
    return max(accuracyList)


path = 'Dataset/'
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
thresholdRange = np.linspace(start=0.40, stop=0.60, num=1000)
for i in range(iterations):
    X_test = np.load('final_test_binarydata_' + str(i) + '.npy')
    Y_test = np.load('final_test_labels_' + str(i) + '.npy')

    # X_test = X_test[:, list_of_gene_numbers]
    X_test = X_test.astype('float')
    X_test = normalize(X_test)
    Y_test = Y_test.astype('float')
    Y_test = Y_test.astype(int)

    with open('Model_ism' + str(i) + '.pkl', 'rb') as f:
        model = pickle.load(f)

    predictionsProb_file = open("predictionsProb_nw_ism" + str(i) + ".csv", 'w')
    predictionProbabilities = model.predict_proba(X_test)

    for prob in predictionProbabilities:
        for pr in prob:
            predictionsProb_file.write(str(pr) + ',')
        predictionsProb_file.write('\n')
    finacclist.append(getResults(predictionProbabilities, Y_test))

print(sum(finacclist)/iterations)
print('Done')