/**
 *
 */
package im.janke.wsdClassifier;

import java.io.File;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * @author Jan Keim
 *
 */
public class EfficientWikiWSDTrainer extends WikiWSDTrainer {

    private String arffDirectory = null;

    /**
     * If this constructor is used it means that the training instances will be thrown away after the first filtering
     * and then loaded one after another to save memory
     *
     * @param arffDirectory
     *            The directory the arff-files are in
     * @param instances
     *            training instances
     */
    public EfficientWikiWSDTrainer(String arffDirectory, Instances instances) {
        super(new EfficientNaiveBayes(), instances);
        this.arffDirectory = arffDirectory;
    }

    public EfficientWikiWSDTrainer(Classifier classifier) {
        super(classifier);
    }

    public EfficientWikiWSDTrainer(Instances instances) {
        super(new EfficientNaiveBayes(), instances);
    }

    public EfficientWikiWSDTrainer(Classifier classifier, Instances instances) {
        super(classifier, instances);
    }

    @Override
    public void buildClassifier() {
        try {
            trainingSet = filter(trainingSet, getFilter());
            if (!trainingSet.attribute(0)
                            .isNominal()) {
                trainingSet = stringToNominalFiltering(trainingSet);
            }

            if (arffDirectory == null) {
                originalClassifier.buildClassifier(trainingSet);
            } else {
                Instances emptySet = new DataSource(trainingSet).getStructure();
                originalClassifier.buildClassifier(emptySet);
                trainClassifierBySlowLoadInstances();
            }

        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
        isBuild = true;
    }

    @Override
    public void updateClassifier(Instances instances) {
        if (!instances.attribute(1)
                      .isNominal()) {
            Trainer.logger.warning("Could not update classifier because instances are not nominal!");
            return;
        }
        ExecutorService executor = Executors.newWorkStealingPool();
        for (Instance instance : trainingSet) {
            executor.execute(() -> {
                try {
                    ((EfficientNaiveBayes) originalClassifier).updateClassifier(instance);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }
        executor.shutdown();
        try {
            executor.awaitTermination(42, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        executor = null;
    }

    private void trainClassifierBySlowLoadInstances() {
        // load arff files and update the classifier
        Instances data = null;
        File dir = new File(arffDirectory);
        for (File file : dir.listFiles()) {
            if (!DataSource.isArff(file.getAbsolutePath())) {
                continue;
            }
            try {
                DataSource source = new DataSource(file.getAbsolutePath());
                data = source.getDataSet();
                data.setClassIndex(0);
                ExecutorService executor = Executors.newWorkStealingPool();
                for (Instance instance : data) {
                    executor.execute(() -> {
                        try {
                            ((EfficientNaiveBayes) originalClassifier).updateClassifier(instance);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    });
                }
                source = null;
                data = null;
                executor.shutdown();
                executor.awaitTermination(100, TimeUnit.MINUTES);
                executor = null;
            } catch (Exception e) {
                // TODO
                e.printStackTrace();
            }
        }
    }

    @Override
    public Optional<Evaluation> evaluate(boolean crossValidation) {
        if (!trainingSet.attribute(1)
                        .isNominal()
                || !trainingSet.attribute(trainingSet.numAttributes() - 1)
                               .isNominal()) {
            // attributes are not nominal, filter them first!
            try {
                trainingSet = filterStringToNominal(trainingSet, "2-last");
            } catch (Exception e) {
                e.printStackTrace();
                return Optional.empty();
            }
        }
        if (isBuild) {
            return (crossValidation) ? evaluateCrossValidationFolds(trainingSet, 10)
                    : evaluateWithInstances(trainingSet);
        } else {
            return Optional.empty();
        }
    }

    @Override
    public Classifier getClassifier() {
        return originalClassifier;
    }

    public void setArffDirectory(String directory) {
        if (!(originalClassifier.getClass()
                                .getName()
                                .equals(EfficientNaiveBayes.class.getName()))) {
            throw new IllegalArgumentException(
                    "Setting arff directory only allowed when classifier is EfficientNaiveBayes!");
        }
        if (new File(directory).isDirectory()) {
            arffDirectory = directory;
        } else {
            throw new IllegalArgumentException("Provided directory is no directory");
        }
    }
}
