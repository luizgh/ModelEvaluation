import numpy

class Predictor:
    def getPredictions(self, logProbabilities):
        """ Return the top 1 predictions, given a set of probabilities: 
            argmax_c P(c | data) """
        return numpy.argmax(logProbabilities, axis=1).tolist()