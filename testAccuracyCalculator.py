import unittest
import numpy
from AccuracyCalculator import *

class testAccuracyCalculator(unittest.TestCase):
    def setUp(self):
        self.target = AccuracyCalculator()
        
    def testAllSuccess(self):
        predictions = [2,2,1,1,3,3]
        labels = predictions
        
        self.assertEquals((1, []), self.target.calc(predictions, labels))
    
    def testSomeErrors(self):
        predictions = [2,2,1,1,3,3,5,5,4,4]
        labels =      [1,2,1,1,3,3,5,5,4,3]
        
        self.assertEquals((0.8, [0, 9]), self.target.calc(predictions, labels))
    
    def testAllErrors(self):
        predictions = numpy.ones(10)
        labels = predictions+1
        
        result = self.target.calc(predictions, labels)
        self.assertEquals((0, numpy.arange(10).tolist()), result)
    
    def testRaisesIfMoreThanOneDimention(self):
        self.assertRaises(ValueError, self.target.calc, numpy.ones(4).resize(2,2), [0, 1, 2, 3] )
        self.assertRaises(ValueError, self.target.calc, [0, 1, 2, 3], numpy.ones(4).resize(2,2) )
    
if __name__ == '__main__':
    unittest.main()

