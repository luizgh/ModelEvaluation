import unittest
from ConfusionMatrix import ConfusionMatrix
import numpy
from TestUtils import *


class testConfusionMatrix(unittest.TestCase):
    def setUp(self):
        self.target = ConfusionMatrix()
        
    def testNoErrors(self):
        predictions = [0,1,2,3,4]
        labels = predictions
        nClasses = 5
        
        result = self.target.generate(predictions, labels, nClasses)
        self.assertEquals(arrayEqualsTo(numpy.zeros((5,5))), result)
    
    def testOneErrorForEach(self):
        predictions = [0,0,1,1,2,2]
        labels =      [1,2,0,2,0,1]
        nClasses = 3
        expected = numpy.asarray([ [0, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 0]])
        
        
        result = self.target.generate(predictions, labels, nClasses)
        self.assertEquals(arrayEqualsTo(expected), result)
    
    def testFewErrors(self):
        predictions = [0,0,1,1,2,2]
        labels =      [1,1,1,1,0,0]
        nClasses = 3
        
        expected = numpy.asarray([ [0, 2, 0],
                                   [0, 0, 0],
                                   [2, 0, 0]])
        
        result = self.target.generate(predictions, labels, nClasses)
        self.assertEquals(arrayEqualsTo(expected), result)

if __name__ == '__main__':
    unittest.main()
