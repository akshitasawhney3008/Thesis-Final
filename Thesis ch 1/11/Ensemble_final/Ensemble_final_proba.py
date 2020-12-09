from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, matthews_corrcoef, f1_score, roc_curve
import numpy as np

iter = 5

def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1>= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)

def getEnsembleProb(listOfNDArrays, weights):
    listOfEnsembles = []
    array0col0 = listOfNDArrays[0][:, 0]
    array0col1 = listOfNDArrays[0][:, 1]
    array1col0 = listOfNDArrays[1][:, 0]
    array1col1 = listOfNDArrays[1][:, 1]
    for weight in weights:
        newcol0 = array0col0 * weight + array1col0 * (1 - weight)
        newcol1 = array0col1 * weight + array1col1 * (1 - weight)
        listOfEnsembles.append(np.column_stack([newcol0, newcol1]))
    return listOfEnsembles

def getPredictions(ensembes_passed):
    prediction_list = []
    count = 0
    count_list = []
    for ensemble in ensembes_passed:
        count = count + 1
        for th in thresholds:
            count_list.append(count)
            prediction_list.append(getPredictionsGivenThreshold(ensemble, th))
    return count_list,prediction_list

def calculate_accuracy(predictions_passed, actual_values):
    accuracy_list = []
    precision_list = []
    recall_list =[]
    auclist = []
    f1list =[]
    for prediction in predictions_passed:
        accuracy_list.append(accuracy_score(actual_values, prediction))
        precision_list.append(precision_score(actual_values, prediction))
        recall_list.append(recall_score(actual_values, prediction))
        fpr, tpr, thresholds = roc_curve(actual_values, prediction, pos_label=1)
        auroc = auc(fpr, tpr)
        auclist.append(auroc)
        f1list.append(f1_score(actual_values, prediction))
    ind = accuracy_list.index(max(accuracy_list))
    return ind, max(accuracy_list), precision_list[ind], recall_list[ind], auclist[ind], f1list[ind]

print("chap1 and chap2")
path = "C://Users//Arushi//PycharmProjects//ThesisChap2//fixedBuckets(10)//"
netAccuracy = 0
netPrecision = 0
netRecall = 0
netAUC = 0
netF1 = 0
weights = np.linspace(0, 1, num=500)
thresholds = np.linspace(0.40, 0.60, num=500)
for i in range(iter):
    file_lasso = open("pp_lr_xgb_ensemble_chap1" + str(i) + ".csv").readlines()
    file_nw = open("pp_ensemble_chap2_chap2(clust)" + str(i) + ".csv").readlines()
    Y_train = np.load(path + 'final_test_labels_' + str(i) + '.npy')
    Y_train = Y_train.astype('float')
    Y_train = Y_train.astype(int)

    list_of_problasso = []
    list_of_probnw =[]
    for j in range(len(file_lasso)):
        list_of_problasso.append(file_lasso[j].rstrip('\n').rstrip(',').split(','))
        list_of_probnw.append(file_nw[j].rstrip('\n').rstrip(',').split(','))
    nd_a = np.array(list_of_problasso).astype(float)
    nd_b = np.array(list_of_probnw).astype(float)
    print('Getting ensembles')
    ensembles = getEnsembleProb([nd_a, nd_b], weights)
    print('Getting preds')
    countlist, predictions = getPredictions(ensembles)
    ind, final_accuracy, precision,recall, aucroc,f1score = calculate_accuracy(predictions, Y_train)
    np.savetxt('pp_ensemble_chap1_chap2' + str(i) + '.csv', ensembles[countlist[ind]-1], delimiter=',')
    netAccuracy += final_accuracy
    netPrecision += precision
    netRecall += recall
    netAUC += aucroc
    netF1 += f1score

    print("Best accuracy for dataset " + str(i) + " = " + str(final_accuracy))
    print("Best precision for dataset " + str(i) + " = " + str(precision))
    print("Best recall for dataset " + str(i) + " = " + str(recall))
    print("Best aucscore for dataset " + str(i) + " = " + str(aucroc))
    print("Best f1 score for dataset " + str(i) + " = " + str(f1score))


print("Avg acc = " + str(netAccuracy/iter))
print("Avg precision = " + str(netPrecision/iter))
print("Avg recall = " + str(netRecall/iter))
print("Avg auc = " + str(netAUC/iter))
print("Avg f1score = " + str(netF1/iter))

