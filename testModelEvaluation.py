import unittest
from ModelEvaluation import *
from TestUtils import *

class testModelEvaluation(unittest.TestCase):
	def testProbabilities(self):
		probs = numpy.asarray( [ [.4, .1, .1],
								 [.2, .3, .5],
								 [.3, .2, .5]])
		actualLogProbs = ModelEvaluation.GetLogProbabilities(probs)	
		result = ModelEvaluation.GetProbabilities(actualLogProbs)
		
		numpy.testing.assert_array_almost_equal(result, probs)
		
	
	def testPredictionActivations(self):
		probs = numpy.asarray( [ [.4, .1, .1],
						 [.2, .3, .5],
						 [.3, .2, .5]])
		
		expected = numpy.asarray([.4, .5, .5])
		self.assertEquals(arrayEqualsTo(expected), ModelEvaluation.GetPredictedClassActivations(probs))

if __name__ == '__main__':
    unittest.main()
