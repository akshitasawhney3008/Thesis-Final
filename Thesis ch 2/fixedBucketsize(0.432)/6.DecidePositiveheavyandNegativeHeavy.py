myfile1 = open("PosGenesWithThreshDiff_20.txt",'r')
myfile1_read = myfile1.readlines()

myfile2 = open('NegGenesWithThreshDiff_20.txt','r')
myfile2_read = myfile2.readlines()

count_positive_heavy = 0
list_of_positive_heavy_itemsets =[]
for lines in myfile1_read:
    line_split = lines.rstrip(' \n').split(': ')[1].split(',')
    if float(line_split[0]) > float(line_split[1]):
        count_positive_heavy = count_positive_heavy + 1
        list_of_positive_heavy_itemsets.append(lines)
    else:
        print(lines)


print("Number of heavy positive: ", count_positive_heavy)
with open('Positive_heavy_itemsets_221f.txt', 'w') as f:
    for item in list_of_positive_heavy_itemsets:
        f.write(item)

count_negative_heavy = 0
list_of_negative_heavy_itemsets = []
for lines in myfile2_read:
    line_split = lines.rstrip(' \n').split(': ')[1].split(',')
    if float(line_split[0]) > float(line_split[1]):
        count_negative_heavy = count_negative_heavy + 1
        list_of_negative_heavy_itemsets.append(lines)

print("Number of heavy negative: ", count_negative_heavy)
with open('Negative_heavy_itemsets_221f.txt', 'w') as f:
    for item in list_of_negative_heavy_itemsets:
        f.write(item)
