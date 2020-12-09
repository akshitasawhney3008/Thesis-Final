my_file1 = open('Positive_heavy_itemsets_221f.txt','r')
file_read1 = my_file1.readlines()
flag = 0
features =[]
f1 = []
f2 = []
for line in file_read1:
    f1 = line.rstrip(' \n').split(':')[0].split(' ')
    features.append(','.join(f1))


my_file1 = open('Negative_heavy_itemsets_221f.txt','r')
file_read1 = my_file1.readlines()
flag = 0

for line in file_read1:
    f2 = line.rstrip(' \n').split(':')[0].split(' ')
    features.append(','.join(f2))


new_k = []
for elem in features:
    if elem not in new_k:
        new_k.append(elem)

features = new_k
print(features)
print(len(features))

FinalFeaturesFile = open('FinalFeaturesFile.txt', 'w')
FinalFeaturesFile.write(';'.join(features))