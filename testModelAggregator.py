import unittest
import numpy
from ModelAggregator import *
from TestUtils import *

class testModelAggregator(unittest.TestCase):
	def setUp(self):
		self.target = ModelAggregator()
		
	def testNoAggregations(self):
		probabilities = numpy.arange(9).reshape(3,3)
		actual = [1,2,3]
		

		listOfIds = [1, 2, 3]
		
		self.assertTupleEqual((arrayEqualsTo(probabilities), actual, listOfIds), self.target.CalculateJointProbability(probabilities,actual, listOfIds))
		
	def testAggregateAllIntoOne(self):
		probabilities = numpy.arange(9).reshape(3,3)
		actual = [1,1,1]
		listOfIds = ['my', 'my', 'my']

		expectedProbs = arrayEqualsTo(numpy.sum(probabilities, axis=0).reshape(-1,3))
		result = self.target.CalculateJointProbability( probabilities,actual, listOfIds)

		self.assertTupleEqual((expectedProbs, [1],['my']), result)
		
		
	def testAggregateIDwithDifferentLabels_shouldThrow(self):
		probabilities = numpy.arange(9).reshape(3,3)
		actual = [1,1,3]
		listOfIds = ['my', 'my', 'my']

		self.assertRaises(ValueError, self.target.CalculateJointProbability, probabilities,actual, listOfIds)
		
	def testAggregateMultiple(self):
		probabilities = numpy.asarray([ [1, 1, 1],
										[3, 3, 3],
										[2, 2, 2],
										[4, 4, 4]])
		actual = [1,2,1,2]
		listOfIds = ['my1', 'my2','my1', 'my2']
		
		expected = arrayEqualsTo(numpy.asarray([[3, 3, 3], [7, 7, 7]]))
		self.assertTupleEqual((expected, [1,2], ['my1', 'my2']), self.target.CalculateJointProbability(probabilities,actual, listOfIds))
	
	def assertTupleEqual(self, t1,t2):
		for i in range(len(t1)):
			self.assertEquals(t1[i],t2[i])
		
if __name__ == '__main__':
	unittest.main()
