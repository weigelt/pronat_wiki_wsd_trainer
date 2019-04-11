/**
 *
 */
package im.janke.wsdClassifier;

import java.util.ArrayList;
import java.util.Random;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Ignore;
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

	// instances with relatively many classes
	private static Instances instances;
	// instances with only 10 classes
	private static Instances instances2;

	@BeforeClass
	public static void createTestInstances() {
		ArrayList<Attribute> attributes = new ArrayList<>();

		// Declare the class attribute (as string attribute)
		attributes.add(new Attribute("wordSense", true));

		// Declare the feature vector
		attributes.add(new Attribute("actualWord", true));
		attributes.add(new Attribute("actualWordPOS", true));
		attributes.add(new Attribute("word-3", true));
		attributes.add(new Attribute("word-3POS", true));
		attributes.add(new Attribute("word-2", true));
		attributes.add(new Attribute("word-2POS", true));
		attributes.add(new Attribute("word-1", true));
		attributes.add(new Attribute("word-1POS", true));
		attributes.add(new Attribute("word+1", true));
		attributes.add(new Attribute("word+1POS", true));
		attributes.add(new Attribute("word+2", true));
		attributes.add(new Attribute("word+2POS", true));
		attributes.add(new Attribute("word+3", true));
		attributes.add(new Attribute("word+3POS", true));
		attributes.add(new Attribute("leftNN", true));
		attributes.add(new Attribute("leftVB", true));
		attributes.add(new Attribute("rightNN", true));
		attributes.add(new Attribute("rightVB", true));

		// Create an empty training set
		// new Instances(relation name, attribute prototype, initial set
		// capacity)
		EfficientWikiWSDTrainerTest.instances = new Instances("WordSenseDisambiguation", attributes, 10);
		EfficientWikiWSDTrainerTest.instances.setClassIndex(0);

		Random rand = new Random();
		int num_classes = EfficientWikiWSDTrainerTest.LIMIT / 4;
		int max_diff_attributes = EfficientWikiWSDTrainerTest.LIMIT / 5;
		for (int i = 0; i < EfficientWikiWSDTrainerTest.LIMIT; i++) {
			Instance instance = new DenseInstance(attributes.size());
			instance.setDataset(EfficientWikiWSDTrainerTest.instances);
			// set the class. Limit to 256 classes
			instance.setValue(0, "" + (i % num_classes));
			for (int j = 1; j < attributes.size(); j++) {
				instance.setValue(j, "" + rand.nextInt(max_diff_attributes));
			}
			instance.setWeight(2);
			EfficientWikiWSDTrainerTest.instances.add(instance);
		}
	}

	@BeforeClass
	public static void createTestInstances2() {
		ArrayList<Attribute> attributes = new ArrayList<>();

		// Declare the class attribute (as string attribute)
		attributes.add(new Attribute("wordSense", true));

		// Declare the feature vector
		attributes.add(new Attribute("actualWord", true));
		attributes.add(new Attribute("actualWordPOS", true));
		attributes.add(new Attribute("word-3", true));
		attributes.add(new Attribute("word-3POS", true));
		attributes.add(new Attribute("word-2", true));
		attributes.add(new Attribute("word-2POS", true));
		attributes.add(new Attribute("word-1", true));
		attributes.add(new Attribute("word-1POS", true));
		attributes.add(new Attribute("word+1", true));
		attributes.add(new Attribute("word+1POS", true));
		attributes.add(new Attribute("word+2", true));
		attributes.add(new Attribute("word+2POS", true));
		attributes.add(new Attribute("word+3", true));
		attributes.add(new Attribute("word+3POS", true));
		attributes.add(new Attribute("leftNN", true));
		attributes.add(new Attribute("leftVB", true));
		attributes.add(new Attribute("rightNN", true));
		attributes.add(new Attribute("rightVB", true));

		// Create an empty training set
		// new Instances(relation name, attribute prototype, initial set
		// capacity)
		EfficientWikiWSDTrainerTest.instances2 = new Instances("WordSenseDisambiguation", attributes, 10);
		EfficientWikiWSDTrainerTest.instances2.setClassIndex(0);

		Random rand = new Random();
		int num_classes = 10;
		int max_diff_attributes = EfficientWikiWSDTrainerTest.LIMIT / 5;
		for (int i = 0; i < EfficientWikiWSDTrainerTest.LIMIT; i++) {
			Instance instance = new DenseInstance(attributes.size());
			instance.setDataset(EfficientWikiWSDTrainerTest.instances2);
			// set the class. Limit to 256 classes
			instance.setValue(0, "" + (i % num_classes));
			for (int j = 1; j < attributes.size(); j++) {
				instance.setValue(j, "" + rand.nextInt(max_diff_attributes));
			}
			instance.setWeight(2);
			EfficientWikiWSDTrainerTest.instances2.add(instance);
		}
	}

	/**
	 * Tests if the efficient Trainer classifies the same as the original trainer.
	 * They are tested on the training data.
	 */
	@Test
	public void testClassificator_onTraining() {
		Instances instancesCopy = new Instances(EfficientWikiWSDTrainerTest.instances);
		Instances testInstances = new Instances(instancesCopy);
		Trainer efficientTrainer = new EfficientWikiWSDTrainer(instancesCopy);
		// make copy that nothing gets destroyed!
		Trainer baseTrainer = new WikiWSDTrainer(new NaiveBayes(), new Instances(instancesCopy));
		efficientTrainer.buildClassifier();
		baseTrainer.buildClassifier();
		ClassifierService classifierEfficient = new ClassifierService(efficientTrainer.getClassifier(), efficientTrainer.getFilter());
		efficientTrainer = null;
		ClassifierService classifierBase = new ClassifierService(baseTrainer.getClassifier(), baseTrainer.getFilter());
		baseTrainer = null;

		for (int i = 0; i < testInstances.size(); i++) {
			Instance instance = testInstances.get(i);
			String test = classifierEfficient.classifyInstance(instance).getClassificationString();
			String base = classifierBase.classifyInstance(instance).getClassificationString();
			String corr = instance.classAttribute().value((int) instance.classValue());
			Assert.assertEquals("Different classification! Correct:" + corr + "-> Test:" + test + " <-> Base:" + base, base, test);
		}
	}

	/**
	 * Tests if the efficient Trainer classifies the same as the original trainer.
	 * They are tested on the training data.
	 */
	@Ignore
	@Test
	public void testClassificator_onTraining2() {
		Instances instancesCopy = new Instances(EfficientWikiWSDTrainerTest.instances2);
		Instances testInstances = new Instances(instancesCopy);
		Trainer efficientTrainer = new EfficientWikiWSDTrainer(instancesCopy);
		// make copy that nothing gets destroyed!
		Trainer baseTrainer = new WikiWSDTrainer(new NaiveBayes(), new Instances(instancesCopy));
		efficientTrainer.buildClassifier();
		baseTrainer.buildClassifier();
		ClassifierService classifierEfficient = new ClassifierService(efficientTrainer.getClassifier(), efficientTrainer.getFilter());
		efficientTrainer = null;
		ClassifierService classifierBase = new ClassifierService(baseTrainer.getClassifier(), baseTrainer.getFilter());
		baseTrainer = null;

		for (int i = 0; i < testInstances.size(); i++) {
			Instance instance = testInstances.get(i);
			String test = classifierEfficient.classifyInstance(instance).getClassificationString();
			String base = classifierBase.classifyInstance(instance).getClassificationString();
			String corr = instance.classAttribute().value((int) instance.classValue());
			Assert.assertEquals("Different classification! Correct:" + corr + "-> Test:" + test + " <-> Base:" + base, base, test);
		}
	}

	/**
	 * Tests whether the underlaying data is somehow destroyed or stays the same.
	 */
	@Test
	public void testClassificator_dataHandling() {
		Instances instancesCopy = new Instances(EfficientWikiWSDTrainerTest.instances);
		int size_orig = instancesCopy.size();
		double[][] doubleArray = new double[size_orig][instancesCopy.numAttributes()];
		for (int i = 0; i < size_orig; i++) {
			doubleArray[i] = instancesCopy.get(i).toDoubleArray();
		}

		Trainer efficientTrainer = new EfficientWikiWSDTrainer(instancesCopy);
		efficientTrainer.buildClassifier();

		// check if sizes stayed the same
		int size_after_outer = instancesCopy.size();
		Assert.assertEquals(size_orig, size_after_outer);
		int size_after_inner = efficientTrainer.getDataSet().size();
		Assert.assertEquals(size_orig, size_after_inner);

		// check if values changed
		for (int i = 0; i < size_orig; i++) {
			Instance after = efficientTrainer.getDataSet().get(i);
			Assert.assertArrayEquals(doubleArray[i], after.toDoubleArray(), 0.001);
		}
	}
}
