import unittest
from Predictor import Predictor
from TestUtils import *

class testPredictor(unittest.TestCase):
	def testAll(self):
		logProbabilities = numpy.asarray([[0.4, 0.45, 0.41],
										  [0.45, 0.4, 0.41],
										  [0.4, 0.41, 0.45]])
		expected = arrayEqualsTo(numpy.asarray([1,0,2]))
		
		target = Predictor()
		
		self.assertEquals(expected, target.getPredictions(logProbabilities))
					
if __name__ == '__main__':
	unittest.main()
