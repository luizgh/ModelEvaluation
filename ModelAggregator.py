import numpy
class ModelAggregator:
    def Aggregate(self, logProbabilities, labels, ids, function=numpy.sum):
        """ Returns the joint log probability of each entry, where an entry
        is defined by the set of all lines with the same id:
        P(c | data) =  prod_i( P(c | data[i] )) """

        uniqueIds = set(ids)
        idsArray = numpy.asarray(ids)
        labelsArray = numpy.asarray(labels)
        
        aggregatedLogProbs = []
        aggregatedLabels = []
        aggregatedIds = []

        for id in uniqueIds:
            aggregatedIds.append(id)
            aggregatedLogProbs.append(function(logProbabilities[idsArray == id], axis=0))
            thisLabels = labelsArray[idsArray==id]
            if (not numpy.alltrue(thisLabels == thisLabels[0])):
                raise ValueError('Error: rows with the same ID should have the same label.')
            
            aggregatedLabels.append(thisLabels[0])
            
        aggregatedLogProbs = numpy.asarray(aggregatedLogProbs)
        return (aggregatedLogProbs, aggregatedLabels, aggregatedIds)


