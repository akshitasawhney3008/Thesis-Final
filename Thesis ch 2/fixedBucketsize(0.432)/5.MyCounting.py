import pickle

with open('myPosMappingDict_221f', 'rb') as f:
    myNegMappingDict = pickle.load(f)
with open('negative_ism_to_genenames_221f.txt') as f:
    itemSets = f.readlines()
positive_count = 311
negative_count = 206

finalCounts = []
idx = 0

for itemSetLine in itemSets:
    print(idx)
    tempItemSetList = itemSetLine.split('#')[0]
    support = itemSetLine.split('#')[1].rstrip('\n').split(': ')[1]
    value1 = (int(support)/negative_count) * 100
    tempItemSetList = tempItemSetList.split()
    listOfIndicesForItemSets = []
    for itemSet in tempItemSetList:
        if itemSet in myNegMappingDict:
            listOfIndicesForItemSets.append(myNegMappingDict[itemSet])
    if len(listOfIndicesForItemSets) != 0:
        intersection = set(listOfIndicesForItemSets[0])
        for s in listOfIndicesForItemSets[1:]:
            intersection.intersection_update(s)
        value2 = (len(intersection)/positive_count) * 100
    else:
        value2 = 0

    thresh = 20
    if abs(value1-value2) >= thresh:
        my_string = ' '.join(tempItemSetList) + ': ' + str(value1) + ',' + str(value2)
        finalCounts.append(my_string)
    idx += 1

with open('NegGenesWithThreshDiff_' + str(thresh) +'.txt', 'w') as f:
    for item in finalCounts:
        f.write("%s\n" % item)
print('Done')
