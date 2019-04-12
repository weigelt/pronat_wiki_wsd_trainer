package im.janke.wsdClassifier;

import java.util.ArrayList;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Jan Keim
 *
 */
public class EfficientWikiWSDTrainerTest {

    private static final int LIMIT = 1024 * 4;

    private static Instances createTestInstances(ArrayList<Attribute> attributes, int num_classes) {
        Instances instances = new Instances("WordSenseDisambiguation", attributes, 10);
        instances.setClassIndex(0);

        Random rand = new Random();

        int max_diff_attributes = EfficientWikiWSDTrainerTest.LIMIT / 5;
        for (int i = 0; i < EfficientWikiWSDTrainerTest.LIMIT; i++) {
            Instance instance = new DenseInstance(attributes.size());
            instance.setDataset(instances);
            // set the class.
            instance.setValue(0, "" + (i % num_classes));
            for (int j = 1; j < attributes.size(); j++) {
                instance.setValue(j, "" + rand.nextInt(max_diff_attributes));
            }
            instance.setWeight(2);
            instances.add(instance);
        }

        return instances;
    }

    private static ArrayList<Attribute> createAttributes() {
        return ClassifierService.getAttributes();
    }

    private void testClassificator(Instances instances) {
        Instances testInstances = new Instances(instances);

        Trainer efficientTrainer = new EfficientWikiWSDTrainer(instances);
        efficientTrainer.buildClassifier();
        ClassifierService classifierEfficient = new ClassifierService(efficientTrainer.getClassifier(),
                efficientTrainer.getFilter());
        efficientTrainer = null;

        Trainer baseTrainer = new WikiWSDTrainer(new NaiveBayes(), new Instances(instances));
        baseTrainer.buildClassifier();
        ClassifierService classifierBase = new ClassifierService(baseTrainer.getClassifier(), baseTrainer.getFilter());
        baseTrainer = null;

        for (int i = 0; i < testInstances.size(); i++) {
            Instance instance = testInstances.get(i);
            Classification testClassification = classifierEfficient.classifyInstance(instance);
            Classification baseClassification = classifierBase.classifyInstance(instance);

            String test = testClassification.getClassificationString();
            String base = baseClassification.getClassificationString();
            String corr = instance.classAttribute()
                                  .value((int) instance.classValue());

            Assert.assertEquals("Different classification for instance " + i + "! Correct was " + corr + ".", base,
                    test);
        }
    }

    /**
     * Tests if the efficient Trainer classifies the same as the original trainer. They are tested on a set of training
     * data with many classes (LIMIT / 4).
     * 
     */
    @Test
    public void testClassificator_highClassCount() {
        ArrayList<Attribute> attributes = createAttributes();
        int numClasses = EfficientWikiWSDTrainerTest.LIMIT / 4;
        Instances instances = createTestInstances(attributes, numClasses);

        testClassificator(instances);
    }

    /**
     * Tests if the efficient Trainer classifies the same as the original trainer. They are tested on a set of training
     * data with only 10 different classes.
     */
    @Test
    public void testClassificator_lowClassCount() {
        ArrayList<Attribute> attributes = createAttributes();
        int numClasses = 10;
        Instances instances = createTestInstances(attributes, numClasses);

        testClassificator(instances);
    }

    /**
     * Tests if the efficient Trainer classifies the same as the original trainer. They are tested on a set of training
     * data with only 10 different classes where the attributes are all specific for one class
     */
    @Test
    public void testClassificator_lowClassCountTrivialAttributes() {
        ArrayList<Attribute> attributes = createAttributes();
        int numClasses = 10;
        Instances instances = new Instances("WordSenseDisambiguation", attributes, 10);
        instances.setClassIndex(0);

        for (int i = 0; i < EfficientWikiWSDTrainerTest.LIMIT; i++) {
            int corrClass = (i % numClasses);
            String classString = String.valueOf(corrClass);
            Instance instance = new DenseInstance(attributes.size());
            instance.setDataset(instances);
            instance.setValue(0, classString);
            for (int j = 1; j < attributes.size(); j++) {
                instance.setValue(j, classString);
            }
            instance.setWeight(2);
            instances.add(instance);
        }

        testClassificator(instances);
    }

    /**
     * Tests whether the underlying data is somehow destroyed or stays the same.
     */
    @Test
    public void testClassificator_dataHandling() {
        ArrayList<Attribute> attributes = createAttributes();
        int numClasses = EfficientWikiWSDTrainerTest.LIMIT / 4;
        Instances instances = createTestInstances(attributes, numClasses);
        int size_orig = instances.size();
        double[][] doubleArray = new double[size_orig][instances.numAttributes()];
        for (int i = 0; i < size_orig; i++) {
            doubleArray[i] = instances.get(i)
                                      .toDoubleArray();
        }

        Trainer efficientTrainer = new EfficientWikiWSDTrainer(instances);
        efficientTrainer.buildClassifier();

        // check if sizes stayed the same
        int size_after_outer = instances.size();
        Assert.assertEquals(size_orig, size_after_outer);
        int size_after_inner = efficientTrainer.getDataSet()
                                               .size();
        Assert.assertEquals(size_orig, size_after_inner);

        // check if values changed
        for (int i = 0; i < size_orig; i++) {
            Instance after = efficientTrainer.getDataSet()
                                             .get(i);
            Assert.assertArrayEquals(doubleArray[i], after.toDoubleArray(), 0.001);
        }
    }
}
