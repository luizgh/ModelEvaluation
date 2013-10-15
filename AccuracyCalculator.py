import numpy
class AccuracyCalculator:
    def calc(self, predictions, labels):
        """ Returns the accuracy of a model, and the misclassified entries."""
        labels = numpy.asarray(labels)
        predictions = numpy.asarray(predictions)
        
        if (len(labels.shape) != 1): raise ValueError('Argument "labels" should have only one dimension')
        if (len(predictions.shape) != 1): raise ValueError('Argument "predictions" should have only one dimension')
        
        acc = sum(predictions == labels) * 1.0 / predictions.shape[0]
        errors = numpy.where(predictions != labels)
        return (acc, errors[0].tolist())