package edu.kit.ipd.parse.wikiWSDTrainer;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.kit.ipd.parse.wikiWSDClassifier.ClassifierService;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Stopwords;

/**
 * This class represents a Trainer, that can handle extracting training data from lines from Wikipedia dumps
 *
 * @author Jan Keim
 *
 */
public class WikiWSDTrainer extends Trainer {
    private StanfordCoreNLP pipeline;
    private Pattern pattern;

    public WikiWSDTrainer(Classifier classifier) {
        super(classifier);
        pipeline = buildPipeline();
        pattern = buildPattern();
    }

    public WikiWSDTrainer(Classifier classifier, Instances data) {
        super(classifier, data);
        pipeline = buildPipeline();
        pattern = buildPattern();
    }

    private StanfordCoreNLP buildPipeline() {
        Properties props = new Properties();
        // problem: ner is slow, but would be helpful
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma"); // parse,dcoref,ner
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        return pipeline;
    }

    private Pattern buildPattern() {
        return Pattern.compile("\\[\\[([\\w\\s\\-\\.\\(\\)']+\\|)?([\\w\\s\\-\\.']+)\\]\\]",
                Pattern.UNICODE_CHARACTER_CLASS);
    }

    /*
     * (non-Javadoc)
     *
     * @see im.janke.wsdTrainer.Trainer#addTrainingData(java.lang.String, java.lang.String)
     */
    @Override
    public synchronized void addTrainingData(String line, String classification) {
        // just call the other method here
        this.addTrainingData(line);
    }

    /*
     * (non-Javadoc)
     *
     * @see im.janke.wsdTrainer.Trainer.Trainer#addTrainingData(java.lang.String)
     */
    @Override
    public void addTrainingData(String line) {
        // prepare sentences and get the actual disambiguations
        Matcher matcher = pattern.matcher(line);

        // make a queue, this way we get the correct order in processing take
        // out one meaning
        // out of the queue, this way the correct meaning is taken
        // key=word, value=Queue<meaning>
        Map<String, ArrayDeque<String>> disambiguations = getDisambiguationMap(matcher);
        if (disambiguations.isEmpty()) {
            return;
        }
        String cleanLine = matcher.replaceAll("$2");

        // run coreNLP on the line
        Annotation document = new Annotation(cleanLine);
        pipeline.annotate(document);

        // String[] frequentWords = this.getMostFrequentWords(document);
        // get sentences and run over them
        for (CoreMap sentence : document.get(SentencesAnnotation.class)) {
            // traverse the tokens in the current sentence
            // save the indices of wanted words for proper usage later
            List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
            List<Integer> senseIndices = getDisambiguationIndices(disambiguations, tokens);
            // copy set of words for the disambiguations
            Set<String> wordSet = new HashSet<>(disambiguations.keySet());

            // go through indices and create the trainingData instances
            Collections.reverse(senseIndices);
            for (Integer index : senseIndices) {
                Instance instance = new DenseInstance(attributes.size());
                instance.setDataset(trainingSet);
                CoreLabel token = tokens.get(index);
                String word = token.get(TextAnnotation.class);
                String wordLemma = token.get(LemmaAnnotation.class);
                if ((word == null) || (wordLemma == null)) {
                    continue;
                }
                String wordPos = token.get(PartOfSpeechAnnotation.class);
                String meaning = disambiguations.get(word.toLowerCase())
                                                .poll();
                if (meaning == null) {
                    // first possibility: word is part of multi-word; treat like
                    // above
                    if (wordPos.startsWith("NN")) {
                        for (String disambiguationPhrase : wordSet) {
                            if (disambiguationPhrase.contains(word.toLowerCase())) {
                                meaning = disambiguations.get(disambiguationPhrase)
                                                         .poll();
                            }
                        }
                    }
                    // check again; skip this index if still null
                    if (meaning == null) {
                        continue;
                    }
                }

                // skip NamedEntities
                if (!word.matches("^[\\p{Ll}].*")) {
                    // does not start with a lowercase letter
                    // -> starts with an upper case (or number)
                    continue;
                }

                synchronized (attributes) {
                    instance.setValue(0, meaning);
                    instance.setValue(1, wordLemma);
                    instance.setValue(2, wordPos);
                }

                // get left and right 3 words along with their POS
                addSurroundingWordsAttributes(tokens, index, instance);

                // add next and previous NN* and VB*
                addLeftAndRightNounsAndVerbs(tokens, index, instance);

                // // add most frequent words
                // for (int i = 0; i < 3; i++) {
                // this.addAttributeToInstance(instance, 19 + i,
                // frequentWords[i]);
                // }

                // finally add instance
                if ((instance != null) && trainingSet.checkInstance(instance)) {
                    // add instance to trainingSet.
                    // synchronized, because underlying structure is a
                    // simple list, that might have problems otherwise
                    // double the weight to have a bigger difference bw laplace
                    // and real instances
                    instance.setWeight(2);
                    synchronized (this) {
                        trainingSet.add(instance);
                    }
                }
            }
        }
    }

    /**
     * Maps the actual disambiguations to the words that represent them
     *
     * @param matcher
     *            Matcher that will be used to find the disambiguations along with their representation
     * @return Mapping of Disambiguation to list of representations for that disambiguation in order of their appearance
     *         within the text.
     */
    private Map<String, ArrayDeque<String>> getDisambiguationMap(Matcher matcher) {
        Map<String, ArrayDeque<String>> disambiguations = new HashMap<>();
        while (matcher.find()) {
            // toLowerCase "accidentally" also removes NamedEntities (when
            // written with capital
            // letter in beginning
            String word = matcher.group(2)
                                 .toLowerCase();
            String meaning = (matcher.group(1) == null) ? word
                    : matcher.group(1)
                             .replace("|", "")
                             .toLowerCase();
            if (meaning.matches("^[\\d]+.*")) {
                // numbers only (e.g. for years) will be omitted
                // also omit smth like "2004 afl championship"
                continue;
            }
            ArrayDeque<String> queue = disambiguations.get(word);
            if (queue != null) {
                queue.offer(meaning);
            } else {
                queue = new ArrayDeque<>();
                queue.offer(meaning);
                disambiguations.put(word, queue);
            }
        }
        return disambiguations;
    }

    /**
     * Creates the List of indices where the words for the disambiguations are.
     *
     * @param disambiguations
     *            Mapping of disambiguation to words
     * @param tokens
     *            Tokens of the sentence
     * @return List of indices where the disambiguation words are within the sentence
     */
    private List<Integer> getDisambiguationIndices(Map<String, ArrayDeque<String>> disambiguations,
            List<CoreLabel> tokens) {
        List<Integer> senseIndices = new ArrayList<>();
        // get the set of words for comparisons of words
        // deep copy it properly!
        Set<String> wordSet = new HashSet<>(disambiguations.keySet());
        for (int i = tokens.size() - 1; i >= 0; i--) {
            CoreLabel token = tokens.get(i);
            String word = token.get(TextAnnotation.class);
            if (disambiguations.containsKey(word.toLowerCase())) {
                senseIndices.add(i);
            } else {
                // for multi-word-stuff
                String wordPos = token.get(PartOfSpeechAnnotation.class);
                // we focus primarily on nouns!
                if (wordPos.startsWith("NN")) {
                    // TODO: make this better!
                    Set<String> wordSetCopy = new HashSet<>(wordSet);
                    outer: for (String disambiguationPhrase : wordSetCopy) {
                        if (disambiguationPhrase.contains(word.toLowerCase())) {
                            // idea one: add the last noun, but in some cases
                            // the first noun of
                            // the phrase
                            int index = i - 1;
                            int senseIndex = i;
                            boolean takeFirst = false;
                            while (index >= 0) {
                                CoreLabel tmpToken = tokens.get(index);
                                String tmpWord = tmpToken.get(TextAnnotation.class);
                                String tmpPos = token.get(PartOfSpeechAnnotation.class);
                                if (disambiguationPhrase.contains(tmpWord.toLowerCase())) {
                                    if (tmpPos.equals("IN") || tmpPos.equals("TO")) {
                                        takeFirst = true;
                                    } else if (takeFirst && tmpPos.startsWith("NN")) {
                                        senseIndex = index;
                                    }
                                    index--;
                                } else {
                                    break;
                                }
                            }
                            // make sure, only once added! (per meaning)
                            for (int existingSenseIndex : senseIndices) {
                                String tmp_word = tokens.get(existingSenseIndex)
                                                        .get(TextAnnotation.class);
                                if (disambiguationPhrase.contains(tmp_word.toLowerCase())) {
                                    continue outer;
                                }
                            }
                            senseIndices.add(senseIndex);

                            // idea two: just add every noun, that is contained
                            // in the phrase,
                            // as in usage every noun might be used to ask for
                            // meaning
                            // senseIndices.add(i);
                            break;
                        }
                    }
                }
            }
        }
        return senseIndices;
    }

    /**
     * Creates the Surrounding Word Attribute Values for the provided instance. Surrounding words are left 3 words and
     * right 3 words, stopwords and unwanted words are filtered. For each word its POS is also evaluated and put into
     * the instance.
     *
     * @param tokens
     *            Tokens of the sentence
     * @param index
     *            Index of the actual disambiguation word
     * @param instance
     *            Instance that should be generated
     */
    private void addSurroundingWordsAttributes(List<CoreLabel> tokens, Integer index, Instance instance) {
        int leftAdd = 0;
        int rightAdd = 0;
        for (int i = 1; i <= 3; i++) {
            // left:
            String leftLemma = "NONE";
            String leftPos = "NONE";
            int leftIndex = index - i;
            if (leftIndex >= 0) {
                CoreLabel leftToken = tokens.get(leftIndex);
                leftLemma = leftToken.get(LemmaAnnotation.class);
                leftLemma = ((leftLemma != null) && (!leftLemma.startsWith("#"))) ? leftLemma.toLowerCase() : "NONE";
                while (Stopwords.isStopword(leftLemma) || ClassifierService.filterWords.contains(leftLemma)) {
                    leftAdd += 1;
                    // when word is a word, that should be filtered, skip it!
                    if ((leftIndex - leftAdd) < 0) {
                        leftLemma = "NONE";
                        break;
                    }
                    leftToken = tokens.get(leftIndex - leftAdd);
                    leftLemma = leftToken.get(LemmaAnnotation.class);
                    leftLemma = ((leftLemma != null) && (!leftLemma.startsWith("#"))) ? leftLemma.toLowerCase()
                            : "NONE";
                }
                leftPos = leftToken.get(PartOfSpeechAnnotation.class);
            }
            // word-3 is at index 3
            // word-1 is at index 7
            // 2*i because we have word-i and wordPos-i
            int attributeIndex = 9 - (2 * i);
            if (!leftLemma.equals("NONE")) {
                synchronized (attributes) {
                    instance.setValue(attributeIndex, leftLemma);
                    instance.setValue(attributeIndex + 1, leftPos);
                }
            }

            // right:
            String rightLemma = "NONE";
            String rightPos = "NONE";
            int rightIndex = index + i;
            if (rightIndex < tokens.size()) {
                CoreLabel rightToken = tokens.get(rightIndex);
                rightLemma = rightToken.get(LemmaAnnotation.class);
                rightLemma = ((rightLemma != null) && (!rightLemma.startsWith("#"))) ? rightLemma.toLowerCase()
                        : "NONE";
                while (Stopwords.isStopword(rightLemma) || ClassifierService.filterWords.contains(rightLemma)) {
                    rightAdd += 1;
                    // when word is a word, that should be filtered, skip it!
                    if ((rightIndex + rightAdd) >= tokens.size()) {
                        rightLemma = "NONE";
                        break;
                    }
                    rightToken = tokens.get(rightIndex + rightAdd);
                    rightLemma = rightToken.get(LemmaAnnotation.class);
                    rightLemma = ((rightLemma != null) && (!rightLemma.startsWith("#"))) ? rightLemma.toLowerCase()
                            : "NONE";
                }
                rightPos = rightToken.get(PartOfSpeechAnnotation.class);
            }
            // word+1 starts at 9
            attributeIndex = 7 + (2 * i);
            if (!rightLemma.equals("NONE")) {
                synchronized (attributes) {
                    instance.setValue(attributeIndex, rightLemma);
                    instance.setValue(attributeIndex + 1, rightPos);
                }
            }
        }
    }

    /**
     * Finds the Noun and Verb that are each left and right next to the actual disambiguation word
     *
     * @param tokens
     *            Tokens of the sentence
     * @param index
     *            Index of the actual disambiguation word
     * @param instance
     *            Instance to be generated
     */
    private void addLeftAndRightNounsAndVerbs(List<CoreLabel> tokens, Integer index, Instance instance) {
        String leftNN = "NONE";
        String leftVB = "NONE";
        String rightNN = "NONE";
        String rightVB = "NONE";
        for (int i = 1; i < tokens.size(); i++) {
            int leftIndex = index - i;
            if (leftIndex >= 0) {
                CoreLabel leftToken = tokens.get(leftIndex);
                String leftPOS = leftToken.get(PartOfSpeechAnnotation.class);
                if (leftNN.equals("NONE") && leftPOS.startsWith("NN")) {
                    // we found left NN*
                    leftNN = leftToken.get(LemmaAnnotation.class);
                    leftNN = ((leftNN != null) && !leftNN.startsWith("#")) ? leftNN.toLowerCase() : "NONE";
                    if (ClassifierService.filterWords.contains(leftNN)) {
                        leftNN = "NONE";
                    }
                } else if (leftVB.equals("NONE") && leftPOS.startsWith("VB")) {
                    // we found left VB*
                    leftVB = leftToken.get(LemmaAnnotation.class);
                    leftVB = ((leftVB != null) && !leftNN.startsWith("#")) ? leftVB.toLowerCase() : "NONE";
                    if (ClassifierService.filterWords.contains(leftVB)) {
                        leftVB = "NONE";
                    }
                }
            }

            int rightIndex = index + i;
            if (rightIndex < tokens.size()) {
                CoreLabel rightToken = tokens.get(rightIndex);
                String rightPOS = rightToken.get(PartOfSpeechAnnotation.class);
                if (rightNN.equals("NONE") && rightPOS.startsWith("NN")) {
                    // we found right NN*
                    rightNN = rightToken.get(LemmaAnnotation.class);
                    rightNN = ((rightNN != null) && !rightNN.startsWith("#")) ? rightNN.toLowerCase() : "NONE";
                    if (ClassifierService.filterWords.contains(rightNN)) {
                        rightNN = "NONE";
                    }
                } else if (rightVB.equals("NONE") && rightPOS.startsWith("VB")) {
                    // we found right VB*
                    rightVB = rightToken.get(LemmaAnnotation.class);
                    rightVB = ((rightVB != null) && !rightNN.startsWith("#")) ? rightVB.toLowerCase() : "NONE";
                    if (ClassifierService.filterWords.contains(rightVB)) {
                        rightVB = "NONE";
                    }
                }
            }

            if (!leftNN.equals("NONE") && !leftVB.equals("NONE") && !rightNN.equals("NONE")
                    && !rightVB.equals("NONE")) {
                break;
            }
        }

        addAttributeToInstance(instance, 15, leftNN);
        addAttributeToInstance(instance, 16, leftVB);
        addAttributeToInstance(instance, 17, rightNN);
        addAttributeToInstance(instance, 18, rightVB);
    }

    /**
     * Adds attribute values to the instance
     *
     * @param instance
     *            Instance
     * @param attrIndex
     *            Index of the attribute
     * @param attrValue
     *            Value to be set
     */
    private void addAttributeToInstance(Instance instance, int attrIndex, String attrValue) {
        if (!attrValue.equals("NONE")) {
            synchronized (attributes) {
                instance.setValue(attrIndex, attrValue);
            }
        }
    }

    @SuppressWarnings("unused")
    private String[] getMostFrequentWords(Annotation document) {
        HashMap<String, Integer> words = new HashMap<>();
        for (CoreLabel token : document.get(TokensAnnotation.class)) {
            String wordLemma = token.get(LemmaAnnotation.class)
                                    .toLowerCase();
            if (Stopwords.isStopword(wordLemma) || ClassifierService.filterWords.contains(wordLemma)
                    || ClassifierService.additionalFilterWords.contains(wordLemma)
                    || wordLemma.matches("^\\+?[\\d]+.*")) {
                // skip stopwords and unwanted words
                // above is based on Rainbow-Stopwords
                // https://www.cs.cmu.edu/~mccallum/bow/rainbow/
                continue;
            }
            Integer counter = words.get(wordLemma);
            if (counter == null) {
                counter = 0;
            }
            counter++;
            words.put(wordLemma, counter);
        }
        List<Entry<String, Integer>> sortedStopwords = new ArrayList<>(words.entrySet());
        Collections.sort(sortedStopwords, (o1, o2) -> o2.getValue() - o1.getValue());
        String[] retStrings = { "NONE", "NONE", "NONE" };
        int counter = 0;
        for (int i = 0; i < sortedStopwords.size(); i++) {
            Entry<String, Integer> entry = sortedStopwords.get(i);
            if (entry.getValue() >= 2) {
                // gets the most frequent word, at least 2 uses
                // if multiple equal frequent, then should return the first one
                // (stable sort)
                retStrings[counter++] = entry.getKey();
            }
            if (counter >= 3) {
                break;
            }
        }

        return retStrings;
    }

    @SuppressWarnings("unused")
    private String lemmatizePOS(String pos) {
        switch (pos) {
        case "JJR":
        case "JJS":
            return "JJ";
        case "NNS":
        case "NNP":
        case "NNPS":
            return "NN";
        case "PRP$":
            return "PRP";
        case "RBR":
        case "RBS":
            return "RB";
        case "VBD":
        case "VBG":
        case "VBN":
        case "VBP":
        case "VBZ":
            return "VB";
        case "WP$":
            return "WP";
        default:
            return pos;
        }
    }

    /*
     * (non-Javadoc)
     *
     * @see im.janke.wsdTrainer.Trainer#buildInstances()
     */
    @Override
    protected void createAttributesAndPrepareInstances() {
        attributes = ClassifierService.getAttributes();
        trainingSet = ClassifierService.getEmptyInstancesHeader(attributes);
    }
}
