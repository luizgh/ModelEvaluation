import unittest
import numpy
from Normalizer import Normalizer
from TestUtils import *

class testNormalizer(unittest.TestCase):
	def testLogProbabilities(self):
		probs = numpy.asarray([ [0.01, 0.04],
								[0.1, 0.4],
								[0.4, 0.1],
								[0.00004, 0.00001]])
		
		expectedProbs =  numpy.asarray([ [0.2, 0.8],
								[0.2, 0.8],
								[0.8, 0.2],
								[0.8, 0.2]])
								
		logProbs = numpy.log(probs)
		logExpectedProbs = numpy.log(expectedProbs)
		
		target = Normalizer()
		
		normalizedLogProbs = target.normalizeLogProbabilities(logProbs)
		normalizedProbs = numpy.exp(normalizedLogProbs)
		
		numpy.testing.assert_array_almost_equal(normalizedProbs, expectedProbs)
		numpy.testing.assert_array_almost_equal(normalizedLogProbs, logExpectedProbs)
		
		
		
if __name__ == '__main__':
    unittest.main()
