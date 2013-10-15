patchPredictions = predictor.getPredictions(patchProbabilities)

(patchAccuracy, patchMisclassified) = accuracyCalculator.Calc(patchPredictions, patchLabels)
patchConfusionMatrix = confusionMatrix.generate(patchPredictions, patchLabels)

(fileProbabilities, fileLabels, fileIdentifiers) = modelAggregator.CalculateJointProbability(patchProbabilities, patchLabels, patchIdentifiers)
filePredictions = getPredictions(fileProbabilities)

(accuracy, misclassified) = accuracyCalculator.calc(filePredictions, fileLabels)

fileConfusionMatrix = confufionMatrix.generate(filePredictions, fileLabels)