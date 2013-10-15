from scipy.misc import logsumexp

class Normalizer:
	def normalizeLogProbabilities(self, array):
		return array - logsumexp(array, axis=1).reshape(-1,1)