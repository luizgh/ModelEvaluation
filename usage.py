nClasses = 10

patchPredictions = ModelEvaluation.GetPredictions(patchProbabilities)

(patchAccuracy, patchMisclassified) = ModelEvaluation.CalculateAccuracy(patchPredictions, patchLabels)
patchConfusionMatrix = ModelEvaluation.GetConfusionMatrix(patchPredictions, patchLabels, nClasses)

predictionActivations = ModelEvaluation.GetPredictedClassActivations(patchPredictions)
correctPredictionActivations = predictionActivations(numpy.asarray(patchPredictions) == patchLabels)
incorrectPredictionActivations = predictionActivations(numpy.asarray(patchPredictions) != patchLabels)


(fileProbabilities, fileLabels, fileIdentifiers) = ModelEvaluation.GetJointLogProbability(patchProbabilities, patchLabels, patchIdentifiers)
filePredictions = ModelEvaluation.GetPredictions(fileProbabilities)

(accuracy, misclassified) = ModelEvaluation.CalculateAccuracy(filePredictions, fileLabels)

fileConfusionMatrix = ModelEvaluation.GetConfusionMatrix(filePredictions, fileLabels)

