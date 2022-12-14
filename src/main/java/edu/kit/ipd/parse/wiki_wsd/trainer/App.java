/**
 *
 */
package edu.kit.ipd.parse.wiki_wsd.trainer;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import edu.kit.ipd.pronat.wiki_wsd.classifier.ClassifierMethod;
import edu.kit.ipd.pronat.wiki_wsd.classifier.ClassifierService;
import edu.kit.ipd.pronat.wiki_wsd.classifier.EfficientNaiveBayes;
import edu.kit.ipd.pronat.wiki_wsd.classifier.SerializationHelper;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Main class of this WSDTrainer.
 *
 * @author Jan Keim
 * @author Sebastian Weigelt
 *
 */
public class App {
	private static final Logger logger = Logger.getLogger(App.class.getName());
	private static final String SUFFIX_CLASSIFIER = ".classifer";
	private static final String SUFFIX_FILTER = ".filter";
	private static final String SUFFIX_INSTANCEHEADER = ".instanceheader";

	// program arguments
	@Option(name = "-a", aliases = "--arff", usage = "Save the training data into the provided filename as arff-file(s). Is input arff if -d is set.")
	private String arffFileName = null;
	@Option(name = "-d", aliases = "--data-provided", usage = "Use the provided arff file as input data.")
	private boolean arffInput = false;
	@Option(name = "-e", aliases = "--evaluate", usage = "Evaluate the classifier after building it")
	private boolean evalClassifier = false;
	@Option(name = "-i", aliases = "--input", usage = "Root Directory the input data files lie in.")
	private String input = null;
	@Option(name = "-n", aliases = "--name", usage = "Output Classifier Name. Default is the name of the classifier followed by '.classifier'")
	private String outputFileName = null;
	@Option(name = "-o", aliases = "--output", usage = "Output Directory. The resulting file will be named after the classifier or the name specified with -n.", required = true)
	private String outputDirectory;
	@Option(name = "-s", aliases = "--split", usage = "Split arff-output into parts, each consisting of the provided amount of articles.")
	private int splitValue = -1;
	@Option(name = "-r", aliases = "--remove-unique", usage = "Remove unique instances before building the classifier.")
	private boolean removeUnique = false;

	private Trainer trainer;
	private int counter = 0;
	private int fileCounter = 0;

	/**
	 * Main method of this program
	 *
	 * @param args
	 *            arguments
	 */
	public static void main(String[] args) {
		App.logger.info("Welcome!");
		new App().start(args);
		App.logger.info("Finished. Bye!");
	}

	/**
	 * Starts the whole process of getting training data and training the
	 * classifier.
	 *
	 * @param args
	 *            arguments
	 */
	private void start(String[] args) {
		processArguments(args);
		logSetParameters();
		Classifier classifier = ClassifierMethod.EfficientNaiveBayes.getClassifier();
		trainer = new EfficientWikiWSDTrainer(classifier);
		// get the files and create or read in the training data
		if (!arffInput) {
			File directory = new File(input);
			if (splitValue > 0) {
				prepareArffSaving();
			}
			logger.info("Start processing input files, saving to arff-file(s) if set.");
			startProcessing(directory);

			if (splitValue > 0) {
				Optional<Instances> instances = getInstancesFromArff();
				if (instances.isPresent()) {
					trainer = new EfficientWikiWSDTrainer(classifier, instances.get());
				}
			}
		} else {
			Optional<Instances> instances = getInstancesFromArff();
			if (instances.isPresent()) {
				trainer = new EfficientWikiWSDTrainer(classifier, instances.get());
			}
		}
		if (!trainer.hasTrainingData()) {
			logger.info("Error! Trainer has no training data! Stopping!");
			return;
		}

		trainer = new EfficientWikiWSDTrainer(trainer.getDataSet());

		// Build classifier and save it
		trainer.setRemoveUnique(removeUnique);
		Instances instancesHeader = new Instances(trainer.getDataSet(), 0);
		logger.info(trainer.dataSummaryString());
		logger.info("Starting to filter instances and build the classifier.");
		trainer.buildClassifier();

		logger.info("Building Classifier finished. Saving it now.");
		save(trainer, instancesHeader);

		// Eval
		if (evalClassifier) {
			miniEval(10000, true);
		}
	}

	/**
	 * Argument processing. Sets needed attributes and checks for problems in
	 * provided arguments or if there were not enough arguments.
	 *
	 * @param args
	 *            arguments
	 */
	private void processArguments(String[] args) {
		CmdLineParser parser = new CmdLineParser(this);
		if (args.length < 1) {
			parser.printUsage(System.out);
			System.exit(-1);
		}
		try {
			parser.parseArgument(args);
			if ((input == null) && (arffFileName == null)) {
				parser.printUsage(System.out);
				System.exit(-1);
			} else if (input != null) {
				if (!input.endsWith("\\")) {
					input += File.separator;
				}
				if (!new File(input).exists()) {
					App.logger.warning("ERROR: Invalid Input Directory: Does not exist!");
					System.exit(-404);
				} else if (!new File(input).isDirectory()) {
					App.logger.warning("ERROR: Input Directory is no directory! Please provide a directory!");
					System.exit(-21);
				}
			}

			// check arff input
			if (arffInput && ((arffFileName == null) || !new File(arffFileName).exists())) {
				App.logger.warning("ERROR: Invalid Arff Input: Is null or does not exist!");
				System.exit(-4);
			}

			// check arff output
			if ((splitValue > 0) && (arffFileName == null)) {
				App.logger.warning("ERROR: Set splitValue but no arff file name (as directory)");
				System.exit(-4);
			}

			// set output file name
			if (outputFileName == null) {
				outputFileName = ClassifierMethod.EfficientNaiveBayes.toString();
			}

			// check output directory
			if (!outputDirectory.endsWith("\\")) {
				outputDirectory += File.separator;
			}
			if (!new File(outputDirectory).exists()) {
				App.logger.warning("ERROR: Invalid Output Directory: Does not exist!");
				System.exit(-1);
			} else if (!new File(outputDirectory).isDirectory()) {
				App.logger.warning("ERROR: Output Directory is no directory! Please provide a directory!");
				System.exit(-21);
			}
		} catch (CmdLineException e) {
			App.logger.warning("ERROR: Unable to parse command-line options: " + e);
			System.exit(-42);
		} catch (SecurityException e) {
			App.logger.warning("ERROR: SecurityException at accessing files: " + e);
			System.exit(-777);
		}
	}

	/**
	 * Starts processing the input data and save the training data if prefered. If a
	 * file is a (txt-)file then this file will be processed line by line and data
	 * will be added as training data.
	 *
	 * @param dir
	 *            directory, that should be traversed or processed
	 * @throws IllegalArgumentException
	 *             when provided file is not a directory
	 */
	// NOTICE: Might be dangerous if a high splitValue is set and files
	// are big, because there might be problems with heap space.
	private void startProcessing(File dir) throws IllegalArgumentException {
		if (!dir.isDirectory()) {
			throw new IllegalArgumentException("Provided File muts be a directory!");
		}

		TrainingDataVisitor trainingDataVisitor = new TrainingDataVisitor();
		try {
			Files.walkFileTree(dir.toPath(), trainingDataVisitor);
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			// shutdown and wait until jobs are done and termination is done
			trainingDataVisitor.executor.shutdown();
			trainingDataVisitor.executor.awaitTermination(60, TimeUnit.MINUTES);
		} catch (InterruptedException | SecurityException e) {
			App.logger.warning(e.toString());
		}
		// finally save the rest.
		if (splitValue > 0) {
			saveTrainingData(true);
		} else if (arffFileName != null) {
			saveTrainingData(false);
		}
	}

	private class TrainingDataVisitor extends SimpleFileVisitor<Path> {
		ExecutorService executor = Executors.newWorkStealingPool();
		private int linesPerWorker = getLinesPerWorker();
		private List<String> lines = new ArrayList<>();

		private int getLinesPerWorker() {
			int processors = Runtime.getRuntime().availableProcessors();
			return (int) Math.ceil((double) splitValue / (double) processors);
		}

		@Override
		public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
			if (attrs.isRegularFile()) {
				counter++;
				try (Stream<String> stream = Files.lines(file)) {
					lines.addAll(stream.filter(line -> line != null).collect(Collectors.toList()));
				} catch (IOException e) {
					App.logger.warning(e.toString());
				}

				if ((splitValue > 0) && ((counter % linesPerWorker) == 0)) {
					List<String> currLines = List.copyOf(lines);
					executor.execute(() -> trainer.addTrainingData(currLines));
					lines = new ArrayList<>();
				}

				if ((splitValue > 0) && (counter >= splitValue)) {
					try {
						executor.shutdown();
						executor.awaitTermination(10, TimeUnit.MINUTES);
					} catch (InterruptedException e) {
						App.logger.warning(e.toString());
						e.printStackTrace();
					}
					saveTrainingData(true);
					counter = 0;
					executor = Executors.newWorkStealingPool();
				}
			}
			return FileVisitResult.CONTINUE;
		}
	}

	private void logSetParameters() {
		StringBuilder infoBuilder = new StringBuilder("Parameter Info:");
		infoBuilder.append("\n Classifier:\t\t\t").append(ClassifierMethod.EfficientNaiveBayes.toString());
		infoBuilder.append("\n Output Directory:\t\t").append(outputDirectory);
		infoBuilder.append("\n Output File Name:\t\t").append(outputFileName);

		if (splitValue > 0) {
			infoBuilder.append("\n Splitting at:\t\t\t").append(splitValue);
		}
		if (arffInput) {
			infoBuilder.append("\n Arff File:\t\t\t").append(arffFileName);
			infoBuilder.append("\n Using Arff input");
		} else {
			infoBuilder.append("\n Input:\t\t\t\t").append(input);
			if (arffFileName != null) {
				infoBuilder.append("\n Arff File:\t\t\t").append(arffFileName);
			}
		}
		if (evalClassifier) {
			infoBuilder.append("\n Classifier will be evaluated");
		}
		if (removeUnique) {
			infoBuilder.append("\n Unique Instances will be removed");
		}
		App.logger.info(infoBuilder.toString());

	}

	private void prepareArffSaving() {
		// prepare saving of training data
		File file = new File(arffFileName);
		if (splitValue > 0) {
			if (!file.exists() || !file.isDirectory()) {
				file.mkdirs();
			}
		} else {
			if ((file.getParentFile() != null) && !file.getParentFile().exists()) {
				// create directories if they are absent
				file.getParentFile().mkdirs();
			}
		}
	}

	/**
	 * Saves the training data. If the boolean is set, then also resets the trainer
	 *
	 * @param resetTrainer
	 *            Reset the trainer if set
	 */
	private synchronized void saveTrainingData(boolean resetTrainer) {
		fileCounter++;
		// only save to file if there actually is training data
		if (trainer.hasTrainingData()) {
			String filename = arffFileName;
			if (splitValue > 0) {
				String extension = getArffExtension();
				filename += File.separator + fileCounter + extension;
			}
			App.logger.info("Trying to save training data to file " + filename);
			try {
				trainer.saveTrainingData(filename);
			} catch (RandomNullPointerException e) {
				// should not occur anymore, still keep it, in case it somehow
				// occurs again
				e.printStackTrace();
				App.logger.warning("Caught a NullPointerException while trying to save the training data.");
				// Remove the created file
				File f = new File(filename);
				if (f.exists()) {
					f.delete();
				}
				long processed = (fileCounter - 1) * splitValue;
				App.logger.warning(
						"Aborting and exiting now. The last filecounter is " + (fileCounter - 1) + ". Try starting with -f " + processed);
				System.exit(-1337);
			}
		} else {
			App.logger.info("Skip saving to file once as there is no training data.");
		}
		// reset trainer if wished
		if (resetTrainer) {
			trainer.resetInstancesAndFilter();
		}
	}

	private String getArffExtension() {
		return ArffLoader.FILE_EXTENSION_COMPRESSED;
	}

	/**
	 * Evaluates the precision of the classifier
	 *
	 * @param limit
	 *            the amount of instances that should be tested
	 * @param unseen
	 *            if set, tests on "limit" amount of instances, that the classifier
	 *            was not trained with.
	 */
	private void miniEval(int limit, boolean unseen) {
		Random rand = new Random();
		// prepare instances
		Instances trainInstances = trainer.getDataSet();

		if (trainInstances.isEmpty()) {
			trainInstances = getInstancesFromArff().orElse(trainInstances);
		}
		limit = Math.min(limit, trainInstances.size());
		Instances testInstances = new Instances(trainInstances, 0);
		for (int i = 0; i < limit; i++) {
			Instance instance;
			int index = rand.nextInt(trainInstances.size());
			if (unseen) {
				instance = trainInstances.get(index);
			} else {
				instance = trainInstances.remove(index);
			}
			testInstances.add(instance);
		}

		if (unseen) {
			// build classifier
			trainer = new EfficientWikiWSDTrainer(trainInstances);
			trainer.buildClassifier();
		}
		ClassifierService classService = new ClassifierService(trainer.getClassifier(), trainer.getFilter());

		// start evaluation
		AtomicInteger correct = new AtomicInteger();
		App.logger.info("Start evaluating");
		Instant start = Instant.now();
		ExecutorService executor = Executors.newWorkStealingPool();
		int i = 0;
		int outputLimit = limit / 20;
		for (Instance instance : testInstances) {
			int index = i++;
			executor.execute(() -> {
				String orig = instance.classAttribute().value((int) instance.classValue());
				// String lemma = instance.stringValue(instance.attribute(1));
				// String cls = classService.classifyInstanceWithLemma(instance, lemma).getClassificationString();
				String cls = classService.classifyInstance(instance).getClassificationString();
				if (cls.equals(orig)) {
					correct.incrementAndGet();
				}
				if ((index % outputLimit) == 0) {
					App.logger.info(String.format("Evaluation at %d%%", (index / outputLimit) * 5));
				}
			});
		}
		executor.shutdown();
		try {
			executor.awaitTermination(1, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		double res = (double) correct.get() / (double) limit;
		Instant end = Instant.now();
		System.out.println(res);
		System.out.println("Duration: " + Duration.between(start, end).getSeconds() + " seconds.");
	}

	/**
	 * Gets the instances from arff file(s)
	 *
	 * @return Optional of instances; might be not set
	 */
	private Optional<Instances> getInstancesFromArff() {
		App.logger.info("Starting to get instances from arff file(s).");
		Instances data = null;
		File dir = new File(arffFileName);
		if (dir.isFile()) {
			// only one file
			try {
				DataSource source = new DataSource(arffFileName);
				data = source.getDataSet();
			} catch (Exception e) {
				errorReadingArff(e);
			}
		} else if (dir.isDirectory()) {
			// directory
			for (File file : dir.listFiles()) {
				if (!DataSource.isArff(file.getAbsolutePath())) {
					continue;
				}
				try {
					DataSource source = new DataSource(file.getAbsolutePath());
					if (data == null) {
						data = source.getDataSet();
						data.setAttributeWeight(1, 10.);
					} else {
						Trainer.mergeInstancesToFirst(data, source.getDataSet());
					}
				} catch (Exception e) {
					errorReadingArff(e);
				}
			}
		} else {
			errorReadingArff(null);
		}

		if ((data != null) && (data.classIndex() == -1)) {
			data.setClassIndex(0);
		}
		return Optional.ofNullable(data);

	}

	private static void errorReadingArff(Exception exception) {
		App.logger.info("Error! Could not load arff file. Aborting!");
		if (exception != null) {
			exception.printStackTrace();
		}
		System.exit(-42);
	}

	/**
	 * Save the classifier
	 */
	private void save(Trainer trainer, Instances instancesHeader) {
		outputFileName = outputDirectory + outputFileName;
		SerializationHelper.serializeEfficientNaiveBayesClassifier((EfficientNaiveBayes) trainer.getClassifier(),
				outputFileName + App.SUFFIX_CLASSIFIER);
		SerializationHelper.serializeFilter(trainer.getFilter(), outputFileName + App.SUFFIX_FILTER);
		Instances header = new Instances(instancesHeader, 0);
		SerializationHelper.serializeInstances(header, outputFileName + App.SUFFIX_INSTANCEHEADER);
	}
}
