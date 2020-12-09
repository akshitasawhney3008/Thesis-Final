def select_differtial_genes(filename1,filename2,threshold):
    list_of_common_genes = []
    if filename1 != 'PositiveEdgeBetweenessCentralities.csv' and filename2 != 'NegativeEdgeBetweenessCentralities.csv':
        list_of_genes_pos = open(filename1, 'r').readline().rstrip('\n').split(',')
        list_of_genes_neg = open(filename2, 'r').readline().rstrip('\n').split(',')
        for gene in list_of_genes_pos:
            geneIndexPos = list_of_genes_pos.index(gene)
            if gene in list_of_genes_neg:
                geneIndexNeg = list_of_genes_neg.index(gene)
                if abs(geneIndexPos-geneIndexNeg) >= threshold:
                    list_of_common_genes.append(gene)

    else:
        threshold = threshold+300
        file_read_pos = open(filename1, 'r').readlines()
        list_of_edges_pos =[]
        for line in file_read_pos:
            list_of_edges_pos.append(tuple(line.rstrip('\n').split(',')))
        file_read_neg = open(filename2, 'r').readlines()
        list_of_edges_neg = []
        for line in file_read_neg:
            list_of_edges_neg.append(tuple(line.rstrip('\n').split(',')))
        for edge in list_of_edges_pos:
            edgeIndexPos = list_of_edges_pos.index(edge)
            if edge in list_of_edges_neg:
                edgeIndexNeg = list_of_edges_neg.index(edge)
            else:
                continue
            if abs(edgeIndexPos-edgeIndexNeg) >= threshold:
                list_of_common_genes.append(edge)

    return list_of_common_genes

list_of_thresholds = [60,70,80,90,100,150,160,170]
writeinfile = open("SelectedFeatures.csv", 'w')
for thresh in list_of_thresholds:
    print(thresh)
    list_of_common_genes_EdgeBetweeness = select_differtial_genes('PositiveEdgeBetweenessCentralities.csv', 'NegativeEdgeBetweenessCentralities.csv', thresh)
    list_of_common_genes_ClosenessCentralities = select_differtial_genes('PositiveClosenessCentralities.csv', 'NegativeClosenessCentralities.csv', thresh)
    list_of_common_genes_CurrentFlowCentralities = select_differtial_genes('PositiveCurrentFlowCentralities.csv', 'NegativeCurrentFlowCentralities.csv', thresh)
    list_of_common_genes_CurrentFlowClosenessCentralities = select_differtial_genes('PositiveCurrentFlowClosenessCentralities.csv', 'NegativeCurrentFlowClosenessCentralities.csv', thresh)
    list_of_common_genes_DegreeCentralities = select_differtial_genes('PositiveDegreeCentralities.csv', 'NegativeDegreeCentralities.csv', thresh)
    list_of_common_genes_get2HopDcGivenNX = select_differtial_genes('Positiveget2HopDcGivenNX.csv', 'Negativeget2HopDcGivenNX.csv', thresh)

    list_of_common_genes = set(list_of_common_genes_ClosenessCentralities) & set(list_of_common_genes_CurrentFlowCentralities) & set(list_of_common_genes_DegreeCentralities)\
                        & set(list_of_common_genes_get2HopDcGivenNX) & set(list_of_common_genes_CurrentFlowClosenessCentralities)

    print(len(list_of_common_genes))
    print(list_of_common_genes)
    writeinfile.write(','.join(list_of_common_genes))
    writeinfile.write('\n')