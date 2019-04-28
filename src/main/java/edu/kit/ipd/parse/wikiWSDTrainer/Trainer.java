package edu.kit.ipd.parse.wikiWSDTrainer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.logging.Logger;

import edu.kit.ipd.parse.wikiWSDClassifier.FilteredClassifierUpdateable;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

/**
 * Abstract class representing a trainer to train a classifier for WSD.
 *
 * @author Jan Keim
 *
 */
// TODO:
// - evaluating is hella slow! Maybe write own (basic) evaluation
// - check options for StringToWordVector: Which to set?
// - maybe use SparseInstance to save memory (in WikiWSDTrainer)
public abstract class Trainer {
    protected static final Logger logger = Logger.getLogger(Trainer.class.getName());
    protected Classifier originalClassifier;
    protected FilteredClassifier fclassifier;
    protected Instances trainingSet;
    private Filter filter;
    protected boolean isBuild = false;

    protected boolean removeUnique = false;

    protected ArrayList<Attribute> attributes;

    public Trainer(Classifier classifier) {
        createAttributesAndPrepareInstances();
        originalClassifier = classifier;
        filter = createPreFilter();
    }

    public Trainer(Classifier classifier, Instances trainingSet) {
        this.trainingSet = trainingSet;
        originalClassifier = classifier;
        filter = createPreFilter();
    }

    private Filter createPreFilter() {
        // StringToNominal Filtering on the fly.
        StringToNominal stringFilter = new StringToNominal();
        stringFilter.setAttributeRange("first-last");
        try {
            stringFilter.setInputFormat(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return stringFilter;
    }

    public void resetInstancesAndFilter() {
        createAttributesAndPrepareInstances();
        filter = createPreFilter();
    }

    /**
     * Processes a line but gives the class, that should be set to the instance(s)
     *
     * @param line
     *            Line that should be processed
     * @param classification
     *            classification of the instance(s)
     */
    public abstract void addTrainingData(String line, String classification);

    /**
     * Processes the line and adds found instances to the training data ({@link #trainingSet}
     *
     * @param line
     *            Line that should be processed
     */
    public abstract void addTrainingData(String line);

    /**
     * Adds the instances to the training set of this trainer.
     *
     * @param instances
     *            instances that should be added to the training data
     */
    public synchronized void addInstancesToTrainingData(Instances instances) {
        try {
            Trainer.mergeInstancesToFirst(trainingSet, instances);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Builds the classifier. Before, do necessary preparation like the StringToNominal-Filtering, if it was not done
     * before. Also creates the classifier (and sets necessary options).
     */
    public void buildClassifier() {
        try {
            trainingSet = filter(trainingSet, getFilter());
            if (!trainingSet.attribute(0)
                            .isNominal()) {
                trainingSet = stringToNominalFiltering(trainingSet);
            }

            prepareClassifier();
            // Build the meta-classifier
            getClassifier().buildClassifier(trainingSet);
            isBuild = true;
        } catch (Exception e) {
            e.printStackTrace();
            isBuild = false;
        }
    }

    private void prepareClassifier() throws Exception {
        // Set necessary infos for the training instances
        trainingSet.setClassIndex(0);

        // Create the FilteredClassifier object
        if (originalClassifier instanceof UpdateableClassifier) {
            fclassifier = new FilteredClassifierUpdateable();
        } else {
            fclassifier = new FilteredClassifier();
        }
        fclassifier.setFilter(getFilter());
        fclassifier.setClassifier(originalClassifier);
    }

    protected Instances stringToNominalFiltering(Instances instances) throws Exception {
        // StringToNominal on the training data before anything else to get
        // nominal classes
        StringToNominal stringFilter = new StringToNominal();
        stringFilter.setAttributeRange("1");
        stringFilter.setInputFormat(instances);
        return Filter.useFilter(instances, stringFilter);
    }

    public Instances filterStringToNominal(Instances instances, String range) throws Exception {
        StringToNominal stringFilter = new StringToNominal();
        stringFilter.setAttributeRange(range);
        stringFilter.setInputFormat(instances);
        return Filter.useFilter(instances, stringFilter);
    }

    public Instances filter(Instances instances, Filter filter) throws Exception {
        if (removeUnique) {
            instances = Trainer.filterOutUniqueInstances(instances);
        }
        return Filter.useFilter(instances, filter);
    }

    /**
     * Filters out unique Instances
     *
     * @param instances
     *            Instances that should be filtered
     * @return filtered instances (list)
     */
    // TODO: maybe make faster!
    public static Instances filterOutUniqueInstances(Instances instances) {
        HashMap<String, List<Instance>> occurenceCounterMap = new HashMap<>();
        for (Instance current : instances) {
            String clazz = current.stringValue(0);
            List<Instance> instList = occurenceCounterMap.get(clazz);
            if (instList == null) {
                instList = new ArrayList<>();
            }
            instList.add(current);
            occurenceCounterMap.put(clazz, instList);
        }

        for (List<Instance> instanceList : occurenceCounterMap.values()) {
            if (instanceList.size() == 1) {
                instances.remove(instanceList.get(0));
            }
        }
        return instances;
    }

    /**
     * Saves the training set to a file denoted by the provided filename.
     *
     * @param filename
     *            the output file name
     * @param filterUnique
     *            filters out unique instances
     */
    public synchronized void saveTrainingData(String filename) throws RandomNullPointerException {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(trainingSet);

        try {
            saver.setFile(new File(filename));
            saver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (NullPointerException e) {
            // should not occur anymore, but still keep it
            saver.cancel();
            try {
                saver.getWriter()
                     .close();
            } catch (IOException e1) {
                e1.printStackTrace();
            }
            throw new RandomNullPointerException(e);
        }
    }

    /**
     * Performs an evaluation with the given training set of instances.
     *
     * @param trainingSet
     *            instances with which should be evaluated
     * @return the evaluation object or null, if there was an exception
     */
    protected Optional<Evaluation> evaluateWithInstances(Instances trainingSet) {
        Evaluation eval = null;
        try {
            eval = new EfficientEvaluation(trainingSet);
            eval.evaluateModel(getClassifier(), trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return Optional.ofNullable(eval);
    }

    /**
     * Evaluates the classifier and returns the {@link weka.classifiers.evaluation.Evaluation}. A 10-fold cross
     * validation is performed.
     *
     * @return Optional of an Evaluation. The Optional is <code>null</code> when there was an exception within the
     *         evaluation or the Trainer was not prepared (use {@link #buildClassifier()} beforehand!)
     */
    public Optional<Evaluation> evaluate() {
        if (isBuild) {
            return this.evaluate(true);
        } else {
            return Optional.empty();
        }
    }

    /**
     * Evaluates the classifier and returns the {@link weka.classifiers.evaluation.Evaluation}. If the provided boolean
     * is set, then a 10-fold cross validation is performed. Otherwise evaluation is done with the training data.
     *
     * @param crossValidation
     *            whether crossValidation should be done
     * @return Optional of an Evaluation. The Optional is <code>null</code> when there was an exception within the
     *         evaluation or the Trainer was not prepared (use {@link #buildClassifier()} beforehand!)
     */
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

    /**
     * Evaluates a cross validation with the specified amount of folds.
     *
     * @param trainingSet
     *            TrainingSet to be used
     * @param folds
     *            number of folds
     * @return the evaluation object or null, if there was an exception
     */
    protected Optional<Evaluation> evaluateCrossValidationFolds(Instances trainingSet, int folds) {
        Evaluation eval = null;
        try {
            eval = new EfficientEvaluation(trainingSet);
            Random rand = new Random(System.currentTimeMillis());
            eval.crossValidateModel(getClassifier(), trainingSet, folds, rand);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return Optional.ofNullable(eval);
    }

    protected abstract void createAttributesAndPrepareInstances();

    public void updateClassifier(Instances instances) {
        if (getClassifier() instanceof UpdateableClassifier) {
            try {
                if (!instances.attribute(0)
                              .isNominal()) {
                    instances = stringToNominalFiltering(instances);
                }
                for (Instance instance : instances) {
                    ((UpdateableClassifier) getClassifier()).updateClassifier(instance);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            throw new UnsupportedOperationException(); // TODO what to do?
        }
    }

    public void updateClassifier(Instance instance) {
        if (getClassifier() instanceof UpdateableClassifier) {
            try {
                ((UpdateableClassifier) getClassifier()).updateClassifier(instance);
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            throw new UnsupportedOperationException();
        }
    }

    /**
     * Returns the classifier
     *
     * @return the classifier
     */
    public Classifier getClassifier() {
        return fclassifier;
        // return originalClassifier;
    }

    /**
     * Returns the filter
     *
     * @return the filter
     */
    public Filter getFilter() {
        return filter;
    }

    /**
     * Returns the data set
     *
     * @return the data set
     */
    public Instances getDataSet() {
        return trainingSet;
    }

    /**
     * @param trainingSet
     *            the trainingSet to set
     */
    public void setDataSet(Instances trainingSet) {
        this.trainingSet = trainingSet;
    }

    /**
     * Returns the attributes as a (Array)List
     *
     * @return the attributes as a (Array)List
     */
    public List<Attribute> getAttributes() {
        return attributes;
    }

    /**
     * Returns true, if the training data has actual instances in it.
     *
     * @return false if there are no instances in the training set. If there are some, returns true
     */
    public boolean hasTrainingData() {
        return trainingSet.size() > 0;
    }

    public boolean isBuild() {
        return isBuild;
    }

    /**
     * @param removeUnique
     *            the removeUnique to set
     */
    public void setRemoveUnique(boolean removeUnique) {
        this.removeUnique = removeUnique;
    }

    public synchronized String dataSummaryString() {
        StringBuffer strBuilder = new StringBuffer(trainingSet.relationName());
        strBuilder.append("\n");
        strBuilder.append("Num Instances:\t");
        strBuilder.append("" + trainingSet.numInstances());
        strBuilder.append("\n");
        strBuilder.append("Num Attributes:\t");
        strBuilder.append("" + trainingSet.numAttributes());
        strBuilder.append("\n");
        strBuilder.append("\n");
        strBuilder.append("   Name               Missing        Unique         Dist\n");
        for (int i = 0; i < trainingSet.numAttributes(); i++) {
            Attribute a = trainingSet.attribute(i);
            AttributeStats as = trainingSet.attributeStats(i);
            strBuilder.append(String.format("%2d ", i));
            strBuilder.append(String.format("%1$-16s", a.name()));
            String s = String.format("%6d / %3d%%  ", as.missingCount,
                    Math.round((100.0 * as.missingCount) / as.totalCount));
            strBuilder.append(s);
            s = String.format("%6d / %3d%%  ", as.uniqueCount, Math.round((100.0 * as.uniqueCount) / as.totalCount));
            strBuilder.append(s);
            strBuilder.append(String.format("%7d", as.distinctCount));
            strBuilder.append("\n");
        }
        return strBuilder.toString();
    }

    /**
     * Merges two instances together, also copying string and nominal attributes. Relation name of the first instance
     * will be taken.
     *
     * See: https://stackoverflow.com/questions/10771558/how-to-merge-two-sets-of-weka-instances-together
     *
     * Use this, not {@link Instances#mergeInstances(Instances, Instances)} because the other one merges only if
     * datasets are the same size and is primarily to merge different attribute values (for kinda same data set)
     *
     * AWARE: Works best, when there are no nominal attributes, only string atrributes. If there are some, then they
     * must be the same for both instances!
     *
     * @param data1
     *            first data instances
     * @param data2
     *            second data instances
     * @return merge of the two provided instances
     * @throws Exception
     *             Exception when reading instances got a problem
     */
    public static Instances mergeInstances(Instances data1, Instances data2) throws Exception {
        // Create a new dataset
        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName());

        Trainer.mergeInstancesToFirst(dest, data2);
        return dest;
    }

    /**
     * Same as {@link #mergeInstances(Instances, Instances)} but this time the second instance gets inserted into the
     * first instance instead of creating a new one.
     *
     * @param data1
     *            first data instances
     * @param data2
     *            second data instances
     * @return merge of the two provided instances into the first one
     * @throws Exception
     *             Exception when reading instances got a problem
     */
    public static void mergeInstancesToFirst(Instances data1, Instances data2) throws Exception {
        // Check where are the string attributes
        int asize = data1.numAttributes();
        boolean strings_pos[] = new boolean[asize];
        for (int i = 0; i < asize; i++) {
            Attribute att = data1.attribute(i);
            strings_pos[i] = ((att.type() == Attribute.STRING) || (att.type() == Attribute.NOMINAL));
        }

        DataSource source = new DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;
        while (source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            data1.add(instance);

            // Copy string attributes
            for (int i = 0; i < asize; i++) {
                if (strings_pos[i]) {
                    data1.instance(data1.numInstances() - 1)
                         .setValue(i, instance.stringValue(i));
                }
            }
        }
    }
}
