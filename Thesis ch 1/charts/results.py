import matplotlib.pylab as plt

plt.rc('font', family='serif')

# SVM-F1
plt.subplot(4, 1, 2)
clf = 'SVM'
metric = 'F1 score'
xlabel = 'experimental scenario'
ylabel = 'f1 score'





orderingList0 = ['Precision','Recall','AUROC','Accuracy','MCC','F1']
# orderingList0 = ['Logistic regression', 'XGBoost', 'Random forest', 'SVM Linear', 'SVM rbf']
valueList0 = [[0.796481955,0.784,0.767806452,0.769343066,0.538908739,0.787952613], [0.757533496,0.778666667,0.734494624,0.738686131,0.476045643,0.764517418],\
             [0.743010967,0.792,0.726645161,0.732846715,0.46490783,0.763401556], [0.764624489,0.808,0.750774194,0.75620438,0.511324909,0.782852826],\
             [0.75160466,0.786666667,0.735268817,0.740145985,0.475792957,0.767641873]]




# orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
#                 'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
#                 'Hold out-CFC']
# valueList1 = [0.502275880589698, 0.502275880589698, 0.502275880589698, 0.479375474726789, 0.465748227350321,
#               0.318213719986702, 0.502275880589698, 0.502275880589698, 0.463596350692207, 0.502275880589698]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
# barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim((0.35, 1))
plt.show()