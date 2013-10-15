import numpy

class ConfusionMatrix:
    def generate (self, predictions, labels, nClasses):
        """ Return a K x K confusion matrix, for K classes. Predicted 
        classes are in rows, actual classes are in cols"""
        
        result = numpy.zeros((nClasses, nClasses))
        for i in range(len(predictions)):
            if (predictions[i] != labels[i]):
                result[predictions[i], labels[i]] +=1
        return result