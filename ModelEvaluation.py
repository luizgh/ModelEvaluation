from ModelAggregator import ModelAggregator
from ConfusionMatrix import ConfusionMatrix
from Normalizer import Normalizer
from Predictor import Predictor
from AccuracyCalculator import AccuracyCalculator
import numpy

class ModelEvaluation:
	@staticmethod
	def CalculateAccuracy(predictions, labels):
		target = AccuracyCalculator()
		return target.calc(predictions,labels)
	
	@staticmethod
	def GetPredictions(logProbabilities):
		target = Predictor()
		return target.getPredictions(logProbabilities)
	
	@staticmethod
	def GetConfusionMatrix(predictions, labels, nClasses):
		target = ConfusionMatrix()
		return target.generate(predictions, labels, nClasses)
	
	@staticmethod
	def GetJointLogProbability(logProbabilities, labels, ids):
		modelAggregator = ModelAggregator()
		normalizer = Normalizer()
		unormalizedLogProbs = modelAggregator.GetUnormalizedJointLogProbability(logProbabilities, labels, ids)
		
		return normalizer.normalizeLogProbabilities(unormalizedLogProbs)
	
	@staticmethod
	def GetUnormalizedJointLogProbability(logProbabilities, labels, ids):
		modelAggregator = ModelAggregator()
		return modelAggregator.GetUnormalizedJointLogProbability(logProbabilities, labels, ids)
		
	@staticmethod
	def GetLogProbabilities(probabilities):
		return numpy.log(probabilities)
		
	@staticmethod
	def GetProbabilities(logProbabilities):
		return numpy.exp(logProbabilities)
	
	@staticmethod
	def GetPredictedClassActivations(logProbabilities):
		return numpy.max(logProbabilities, axis=1)
	