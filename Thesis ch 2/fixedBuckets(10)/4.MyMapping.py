import pickle

with open('negative_genenames_221f.txt') as f:
    lines = f.readlines()

myNegMappingDict = dict()
idx = 0
for line in lines:
    print(idx)
    tempList = line.split()
    for elt in tempList:
        if elt in myNegMappingDict:
            extractedList = myNegMappingDict[elt]
            extractedList.append(idx)
            myNegMappingDict[elt] = extractedList
        else:
            myNegMappingDict[elt] = [idx]
    idx += 1

with open('myNegMappingDict_221f', 'wb') as f:
    pickle.dump(myNegMappingDict, f)
print('Done')
