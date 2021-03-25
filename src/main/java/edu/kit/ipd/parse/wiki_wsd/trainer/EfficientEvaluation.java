package edu.kit.ipd.parse.wiki_wsd.trainer;

import java.io.Serial;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.BatchPredictor;
import weka.core.Instances;

public class EfficientEvaluation extends Evaluation {

	@Serial
	private static final long serialVersionUID = -5377637741727845563L;
	private static final Logger logger = Logger.getLogger(EfficientEvaluation.class.getName());

	public EfficientEvaluation(Instances data) throws Exception {
		super(data);
	}

	public EfficientEvaluation(Instances data, CostMatrix costMatrix) throws Exception {
		super(data, costMatrix);
	}

	@Override
	public void crossValidateModel(Classifier classifier, Instances data, int numFolds, Random random, Object... forPredictionsPrinting)
			throws Exception {
		m_NumFolds = numFolds;

		// Make a copy of the data we can reorder
		// data = new Instances(data); //TODO need a copy or not?
		data.randomize(random);
		if (data.classAttribute().isNominal()) {
			data.stratify(numFolds);
		}

		// We assume that the first element is a
		// weka.classifiers.evaluation.output.prediction.AbstractOutput object
		AbstractOutput classificationOutput = null;
		if (forPredictionsPrinting.length > 0) {
			// print the header first
			classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
			classificationOutput.setHeader(data);
			classificationOutput.printHeader();
		}

		// Do the folds
		for (int i = 0; i < numFolds; i++) {
			EfficientEvaluation.logger.info("Processing fold " + i);
			Instances train = data.trainCV(numFolds, i, random);
			setPriors(train);
			Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
			EfficientEvaluation.logger.info("Building classifier for fold " + i + " with " + train.size() + " instances.");
			copiedClassifier.buildClassifier(train);
			Instances test = data.testCV(numFolds, i);
			EfficientEvaluation.logger.info("Start evaluating fold " + i + " with " + test.size() + " instances.");
			evaluateModel(copiedClassifier, test, forPredictionsPrinting);
		}

		if (classificationOutput != null) {
			classificationOutput.printFooter();
		}
	}

	@Override
	public double[] evaluateModel(Classifier classifier, Instances data, Object... forPredictionsPrinting) throws Exception {
		// for predictions printing
		AbstractOutput classificationOutput = null;

		double predictions[] = new double[data.numInstances()];

		if (forPredictionsPrinting.length > 0) {
			classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
		}

		// NaiveBayes is BatchPredictor but not
		// implementsMoreEfficientBatchPrediction!
		if ((classifier instanceof BatchPredictor) && ((BatchPredictor) classifier).implementsMoreEfficientBatchPrediction()) {
			// make a copy and set the class to missing
			Instances dataPred = new Instances(data);
			for (int i = 0; i < data.numInstances(); i++) {
				dataPred.instance(i).setClassMissing();
			}
			double[][] preds = ((BatchPredictor) classifier).distributionsForInstances(dataPred);
			for (int i = 0; i < data.numInstances(); i++) {
				double[] p = preds[i];

				predictions[i] = evaluationForSingleInstance(p, data.instance(i), true);

				if (classificationOutput != null) {
					classificationOutput.printClassification(p, data.instance(i), i);
				}
			}
		} else {
			// Need to be able to collect predictions if appropriate (for AUC)
			ExecutorService executor = Executors.newWorkStealingPool();
			for (int i = 0; i < data.numInstances(); i++) {
				final int index = i;
				executor.execute(() -> {
					try {
						predictions[index] = evaluateModelOnceAndRecordPrediction(classifier, data.instance(index));
						// if (classificationOutput != null) {
						// classificationOutput.printClassification(classifier,
						// data.instance(i),
						// i);
						// }
					} catch (Exception e) {
						e.printStackTrace(); // TODO ?
					}
				});
			}
			executor.shutdown();
			try {
				executor.awaitTermination(24, TimeUnit.HOURS);
			} catch (InterruptedException e) {
				e.printStackTrace(); // TODO ?
			}
		}

		return predictions;
	}

}
