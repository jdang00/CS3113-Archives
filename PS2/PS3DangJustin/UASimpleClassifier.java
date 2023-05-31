import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.stream.Collectors;

/********************************
 Name:          Justin A. Dang
 Username:      ua711
 Problem Set:   PS2
 Due Date:      23 Feb 2023
 ********************************/

public class UASimpleClassifier {

    HashMap<String, HashMap<String, Double>> NBModel = new HashMap<>();
    ArrayList<String> modelClasses = new ArrayList<>();
    int dataSetSize;
    int featureSetCount;
    int classifierSpecifier = 5;

    // For given classification data set
    HashMap<String, Double> MainProb = new HashMap<>();
    double largeProb = 0;

    List<Integer> continuousArgs = Arrays.asList(3, 4);

    public static void main(String[] args) {



        UASimpleClassifier ai = new UASimpleClassifier();

        ai.train(args[0]);
        ai.printCLI(args);
        ai.test(args[1]);
        ai.predict(args[2]);


    }

    public static double pdf(double x) {
        return Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
    }

    // return pdf(x, mu, sigma) = Gaussian pdf with mean mu and stddev sigma
    public static double pdf(double x, double mu, double sigma) {
        return pdf((x - mu) / sigma) / sigma;
    }

    // return cdf(z) = standard Gaussian cdf using Taylor approximation
    public static double cdf(double z) {
        if (z < -8.0) return 0.0;
        if (z > 8.0) return 1.0;
        double sum = 0.0, term = z;
        for (int i = 3; sum + term != sum; i += 2) {
            sum = sum + term;
            term = term * z * z / i;
        }
        return 0.5 + sum * pdf(z);
    }

    // return cdf(z, mu, sigma) = Gaussian cdf with mean mu and stddev sigma
    public static double cdf(double z, double mu, double sigma) {
        return cdf((z - mu) / sigma);
    }

    public void printCLI(String[] args) {
        System.out.print("************************************************************\n");
        System.out.print("Problem Set:  Problem Set 3:  Naive Bayes Algorithm\n");
        System.out.print("Name:         Justin Dang\n");
        System.out.printf("Synax:        java UASimpleClassifier\t " + args[0] + "\t" + args[1] + "\t" + args[2] + "\n");
        System.out.print("************************************************************\n");
        System.out.print("\n");
        System.out.printf("Training Phase:\t/" + args[0] + "\n");
        System.out.print("--------------------------------------------------------------\n");
        System.out.printf("\t%s %-30s %5d %n", "=>", "Number of Entries(n):", dataSetSize);
        System.out.printf("\t%s %-30s %5d %n", "=>", "Number of Features(p):", featureSetCount - 1);
        System.out.printf("\t%s %-30s %5d %n", "=>", "Number of Distinct Classes(y):", distinctCounter(modelClasses).size());
        System.out.println();
        System.out.println();
    }

    public void printModel() {


        NBModel.forEach((key, value) -> System.out.println(key + " => " + value));

    }

    /**
     * @param filename path to training data set
     */
    public void train(String filename) {

        readFile(filename);

    }

    /**
     * Processes training data to build a supervises machine learning model
     *
     * @param filename path to training data set
     */
    public void readFile(String filename) {

        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));
            String line;
            int trainCounter = 0;
            ArrayList<ArrayList<String>> featureCount = new ArrayList<>();
            String featureName = br.readLine(); // read feature category name in first row
            String[] fLength = featureName.split(",");
            featureSetCount = fLength.length;

            // Create an array list for every feature
            for (int i = 0; i < featureSetCount; i++) {
                featureCount.add(new ArrayList<>());
            }

            //classSet contains a full set of all classes in the training data
            ArrayList<String> classSet = new ArrayList<>();

            while ((line = br.readLine()) != null) {

                String[] featureSet = line.split(",");

                // Add all feature values per column
                for (int i = 0; i < featureSetCount; i++) {
                    if (!continuousArgs.contains(i)) {
                        featureCount.get(i).add(fLength[i] + featureSet[i]);
                    } else {
                        featureCount.get(i).add(featureSet[i]);
                    }

                    if (i == classifierSpecifier) {
                        classSet.add(fLength[i] + featureSet[i]);
                    }
                }

                trainCounter++;
            }

            dataSetSize = trainCounter;

            // Reduce the model set into distinct values
            ArrayList<ArrayList<String>> modelSet = reduceDataSet(featureCount);
            // Build a hashmap of how many times a class appears in the overall data set
            HashMap<String, Double> mapClass = new HashMap<>();
            for (String r : modelClasses) {
                mapClass.put(r, freqCount(r, classSet));
            }


            MainProb = mapClass;

            //Builds classification model
            populateNBModel(mapClass, modelSet, featureCount, classSet, fLength);

            br.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }


    }

    /**
     * Calculate and store class conditional probabilities and CDFs in the main AI model.
     * Four loops are necessary to cover class, feature, value, and data set frequency considerations.
     *
     * @param mapClass     hash table with classes and class occurrences in the data set
     * @param modelSet     reduced data set that lists distinct values found in every feature
     * @param featureCount full data set for frequency search
     * @param classSet     full set of class data for reference
     */
    public void populateNBModel(HashMap<String, Double> mapClass, ArrayList<ArrayList<String>> modelSet, ArrayList<ArrayList<String>> featureCount, ArrayList<String> classSet, String[] fLength) {

        for (String u : mapClass.keySet()) { // per class
            HashMap<String, Double> featureProbability = new HashMap<>();
            int countIterator = 0;
            for (int i = 0; i < modelSet.size(); i++) { // per feature

                if (!continuousArgs.contains(i)) {
                    for (int j = 0; j < modelSet.get(i).size(); j++) { // per value
                        int featureAdd = 0;
                        for (int k = 0; k < featureCount.get(i).size(); k++) { // find frequency
                            if (featureCount.get(i).get(k).equals(modelSet.get(i).get(j)) && classSet.get(k).equals(u)) {
                                featureAdd++;
                            }
                        }
                        featureProbability.put(modelSet.get(i).get(j), featureAdd / mapClass.get(u));
                    }
                } else {
                    HashMap<String, Double> contCalc = buildCDF(featureCount, mapClass, classSet, u, countIterator);
                    for (String r : contCalc.keySet()) {
                        featureProbability.put(fLength[i] + r, contCalc.get(r));
                    }
                    countIterator++;
                }

            }


            NBModel.put(u, featureProbability);


        }

    }

    public HashMap<String, Double> buildCDF(ArrayList<ArrayList<String>> featureCount, HashMap<String, Double> mapClass, ArrayList<String> classSet, String u, int colPoint) {

        HashMap<String, Double> contProbability = new HashMap<>();
        double sum = 0;
        double mean;
        for (int j = 0; j < dataSetSize; j++) {
            if (classSet.get(j).equals(u)) {
                sum = sum + Double.parseDouble(featureCount.get(continuousArgs.get(colPoint)).get(j));
            }
        }
        mean = sum / mapClass.get(u);
        contProbability.put("Mean", mean);

        double varTop = 0;
        for (int j = 0; j < dataSetSize; j++) {
            if (classSet.get(j).equals(u)) {
                double numExpression = (Double.parseDouble(featureCount.get(continuousArgs.get(colPoint)).get(j))) - contProbability.get("Mean");
                varTop = varTop + Math.pow(numExpression, 2);
            }

        }

        double sd = Math.sqrt(varTop / mapClass.get(u));
        contProbability.put("SD", sd);


        return contProbability;
    }

    /**
     * Counts how many unique values in a discrete feature for a whole data set.
     * Ignore sets in the full data set given in the continuousArgs
     *
     * @param fullData the full data set given as an ArrayList of ArrayLists containing Strings
     * @return the same data structure as the given model but now populated with only one occurrence of each distinct value
     */
    public ArrayList<ArrayList<String>> reduceDataSet(ArrayList<ArrayList<String>> fullData) {

        ArrayList<ArrayList<String>> reducedSet = new ArrayList<>();

        for (int i = 0; i < fullData.size(); i++) {

            // Do not reduce set for continuous vales
            if (!continuousArgs.contains(i)) {
                List<String> uniqueCol = distinctCounter(fullData.get(i));
                ArrayList<String> alConvert = new ArrayList<>(uniqueCol);

                if (i == classifierSpecifier) {
                    modelClasses = alConvert;
                } else {
                    reducedSet.add(alConvert);
                }
            } else {
                reducedSet.add(new ArrayList<>(fullData.get(i)));

            }
        }

        return reducedSet;

    }

    /**
     * Counts the amount of distinct values in any given feature. Used by reduceDataSet() to build a reduced model
     *
     * @param feature an ArrayList of discrete String features
     * @return a List of unique occurrences in the given feature set
     */
    public List<String> distinctCounter(ArrayList<String> feature) {

        return feature.stream().distinct().collect(Collectors.toList());

    }


    /**
     * The number of times a value appears in a set of data
     *
     * @param value   given value
     * @param fullSet given set of data
     * @return integer of the amount of times the value was found in the data set
     */
    public double freqCount(String value, ArrayList<String> fullSet) {

        return Collections.frequency(fullSet, value);
    }


    public void test(String filename) {

        System.out.printf("%s%n", "Testing phase:");
        System.out.print("--------------------------------------------------------------\n");
        System.out.println();

        System.out.printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n", "F1", "F2", "F3", "F4", "F5", "CLASS", "PREDICT", "PROB", "RESULT");
        System.out.printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s \n", "---", "---", "---", "-----", "-------", "-----", "-------", "-----", "---------");

        int incorrect = 0;
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));

            String line;

            br.readLine();
            int testSetCount = 0;
            while ((line = br.readLine()) != null) {

                String[] d = line.split(",");
                String prediction = classify(d[0], d[1], d[2], Double.parseDouble(d[3]), Double.parseDouble(d[4]));
                String result = prediction.equals(d[5]) ? "CORRECT" : "INCORRECT";
                if (!prediction.equals(d[5])) {
                    incorrect++;
                }
                System.out.printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %.1f%% %-4s %-10s \n", d[0], d[1], d[2], d[3], d[4], "class" + d[5], "class" + prediction, largeProb*100,"", result);
                testSetCount++;
            }

            dataSetSize = testSetCount;

            br.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        double percentRight = ((dataSetSize - incorrect) * 100);


        System.out.println();

        System.out.printf("\t%s\t%d%s%d%s %.2f%% %s%n%n", "Total Accuracy:", (dataSetSize - incorrect), " correct / ", dataSetSize, " total\t= ", percentRight / dataSetSize, " Accuracy");
        System.out.printf("\t%s %10d%n","=> Number of Entries(n)", dataSetSize);


    }

    public String classify(String f1, String f2, String f3, double f4, double f5) {

        double prob0 = (MainProb.get("Exited0") / dataSetSize) * (NBModel.get("Exited0").get("Geography" + f1) *
                NBModel.get("Exited0").get("IsActiveMember" + f2) * NBModel.get("Exited0").get("HasCrCard" + f3) *
                cdf(f4, NBModel.get("Exited0").get("BalanceMean"), NBModel.get("Exited0").get("BalanceSD")) *
                cdf(f5, NBModel.get("Exited0").get("CreditScoreMean"), NBModel.get("Exited0").get("CreditScoreSD")));
        double prob1 = (MainProb.get("Exited1") / dataSetSize) * (NBModel.get("Exited1").get("Geography" + f1) *
                NBModel.get("Exited1").get("IsActiveMember" + f2) * NBModel.get("Exited1").get("HasCrCard" + f3) *
                cdf(f4, NBModel.get("Exited1").get("BalanceMean"), NBModel.get("Exited1").get("BalanceSD")) *
                cdf(f5, NBModel.get("Exited1").get("CreditScoreMean"), NBModel.get("Exited1").get("CreditScoreSD")));

        double argMax0 = (prob0) / (prob0 + prob1);
        double argMax1 = (prob1) / (prob0 + prob1);
        largeProb = argMax0 > argMax1 ? argMax0 : argMax1;

        return argMax0 > argMax1 ? "0" : "1";

    }

    public void predict(String filename){
        System.out.printf("\n");
        System.out.printf("Prediction Phase:\n");
        System.out.print("--------------------------------------------------------------\n");
        System.out.printf("\n");

        System.out.printf("\t%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n", "F1", "F2", "F3", "F4", "F5", "CLASS", "PREDICT", "PROB", "RESULT");
        System.out.printf("\t%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s \n", "---", "---", "---", "-----", "-------", "-----", "-------", "-----", "---------");

        int incorrect = 0;
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));

            String line;

            br.readLine();
            int testSetCount = 0;
            while ((line = br.readLine()) != null) {

                String[] d = line.split(",");
                String prediction = classify(d[0], d[1], d[2], Double.parseDouble(d[3]), Double.parseDouble(d[4]));
                String result = prediction.equals(d[5]) ? "CORRECT" : "INCORRECT";
                if (!prediction.equals(d[5])) {
                    incorrect++;
                }
                System.out.printf("\t%-10s %-10s %-10s %-10s %-10s %-10s %-10s %.1f%% %-4s %-10s \n", d[0], d[1], d[2], d[3], d[4], "class" + d[5], "class" + prediction, largeProb*100,"", result);
                testSetCount++;
            }

            dataSetSize = testSetCount;

            br.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }


        System.out.println();
        System.out.printf("\t\t%s %10d","=> Number of Entries(n)", dataSetSize);

    }

}
