/********************************
 Name:          Justin Dang
 Username:      ua711
 Problem Set:    PS3
 Due Date:      9 March 2023
 ********************************/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;


public class PS3 {

    ArrayList<ArrayList<Double>> LRModel = new ArrayList<>();
    ArrayList<ArrayList<Double>> TModel = new ArrayList<>();

    ArrayList<Double> modelWeights = new ArrayList<>();

    ArrayList<Double> targetFeature = new ArrayList<>();
    ArrayList<Double> targetFeatureNotNormal = new ArrayList<>();
    ArrayList<Double> losses = new ArrayList<>();

    double lossValue = 0;

    double featureCount = 0;
    double recordCount = 1;
    double trainingThreshold = 0.8;
    ArrayList<Double> means = new ArrayList<>();
    ArrayList<Double> sds = new ArrayList<>();

    double targetMean = 0;
    double targetSD = 0;

    public static void main(String[] args) {

        String input = args[0];
        int target = Integer.parseInt(args[1]) + 1;
        String output = args[2];
        double learningRate = Double.parseDouble(args[3]);
        double error = Double.parseDouble(args[4]);

        System.out.println("************************************************************\n" +
                "Problem Set:  Problem Set 4:  Gradient Descent\n" +
                "Name:         Justin Dang\n" +
                "Syntax:       java PS3 " + args[0] + " " + args[1] + " " + args[2] + " " + " " + args[3] + " " + args[4] +
                "\n************************************************************\n");


        PS3 ai = new PS3();
        System.out.printf("%s\n", "Training Phase:" + args[0]);
        System.out.print("--------------------------------------------------------------\n");
        ai.readFile(input, ai.trainingThreshold, target);
        System.out.printf("\t=> %s:\t%.0f\n\t=> %s:\t%.0f\n", "Number of Entries(n)", ai.recordCount, "Number of Features(p)", ai.featureCount + 1);
        ai.batchGradientDescent(learningRate, error, ai.trainingThreshold);
        ai.testing(target);
        ai.writeFile(output);
        ai.graphMaker();


    }

    public void writeFile(String filename) {


        try (FileWriter fw = new FileWriter(filename)) {

            for(double v : modelWeights){
                fw.write(v+"\n");
            }

            fw.flush();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }



    }

    public void testing(int feature) {

        System.out.println("\nTesting Phase:");
        System.out.println("--------------------------------------------------------------");

        LRModel = TModel;
        recordCount = LRModel.get(1).size();
        for (int i = 0; i < LRModel.get(0).size(); i++) {
            targetFeatureNotNormal.add(LRModel.get(feature - 1).get(i));
        }

        for (int i = 0; i < featureCount - 1; i++) {

            ArrayList<Double> norm = normalizeData(i, 1, "testing", feature);

            LRModel.get(i).clear();
            LRModel.get(i).addAll(norm);

        }
        addBias(1, feature);
        transposeModel(1);


        double error = 0;

        // Output asks for total loss of data on top. This method calculates it.
        for (int i = 0; i < recordCount; i++) {
            double predictedValue = 0;
            for (int j = 0; j < featureCount; j++) {
                predictedValue = predictedValue + (modelWeights.get(j) * LRModel.get(i).get(j));
            }
            predictedValue = predictedValue * targetSD + targetMean;
            error = error + Math.abs(targetFeatureNotNormal.get(i) - predictedValue);

        }

        System.out.printf("\tLoss of Testing Data %10.3f\n\n", error);

        for (int i = 0; i < recordCount; i++) {
            double predictedValue = 0;
            System.out.printf("\tTest Record: %-5d", i + 1);
            for (int j = 0; j < featureCount; j++) {
                predictedValue = predictedValue + (modelWeights.get(j) * LRModel.get(i).get(j));
            }

            predictedValue = predictedValue * targetSD + targetMean;
            System.out.printf("True: %-10.2f", targetFeatureNotNormal.get(i));
            System.out.printf("Prediction: %-10.3f", predictedValue);
            error = Math.abs(targetFeatureNotNormal.get(i) - predictedValue);
            System.out.printf("Error: %.3f\n", error);


        }

        System.out.printf("\n\t=> Number of Test Entries (n): %10.0f", recordCount);


    }

    private void addBias(double threshold, int feature) {
        ArrayList<Double> biasTerm = new ArrayList<>();

        for (int i = 0; i < recordCount * threshold; i++) {
            biasTerm.add(1.0);
        }


        LRModel.add(0, biasTerm);

        targetFeature = LRModel.get(feature);
        LRModel.remove(feature);
    }


    public void readFile(String filename, double threshold, int target) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));
            String line;
            String[] setData = br.readLine().split(",");

            featureCount = setData.length;

            // Build List for distinct features in model
            for (String y : setData) {
                ArrayList<Double> vectorSplit = new ArrayList<>();
                vectorSplit.add(Double.parseDouble(y));
                LRModel.add(vectorSplit);
            }

            // Populate lists
            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");
                List<String> stringVector = Arrays.asList(data);

                List<Double> rowVector = stringVector.stream()
                        .map(Double::parseDouble)
                        .toList();

                for (int i = 0; i < featureCount; i++) {
                    LRModel.get(i).add(rowVector.get(i));
                }
                recordCount++;

            }

            br.close();

            for (int i = 0; i < featureCount; i++) {
                ArrayList<Double> temp = new ArrayList<>();
                for (int j = (int) (recordCount - (recordCount * (1 - threshold))); j < recordCount; j++) {
                    temp.add(LRModel.get(i).get(j));
                }
                TModel.add(temp);
            }


            for (int i = 0; i < featureCount; i++) {

                ArrayList<Double> norm = normalizeData(i, threshold, "training", target);

                LRModel.get(i).clear();
                LRModel.get(i).addAll(norm);
            }

            addBias(threshold, target);


        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }

    public ArrayList<Double> normalizeData(int feature, double threshold, String type, int target) {

        ArrayList<Double> normValues = new ArrayList<>();

        double mu;
        if (Objects.equals(type, "training")) {

            mu = findMu(feature, threshold);
            double sigma = findSigma(feature, mu, threshold);
            if (feature == target - 1) {
                targetMean = mu;
                targetSD = sigma;
            } else {
                means.add(mu);
                sds.add(sigma);
            }


            for (int i = 0; i < recordCount * threshold; i++) {
                normValues.add((LRModel.get(feature).get(i) - mu) / sigma);
            }
        } else {

            mu = means.get(feature);
            double sigma = sds.get(feature);

            for (int i = 0; i < recordCount * threshold; i++) {
                normValues.add((LRModel.get(feature).get(i) - mu) / sigma);
            }


        }
        return normValues;

    }

    public double findMu(int feature, double threshold) {

        double sum = 0;

        for (int i = 0; i < recordCount * threshold; i++) {
            sum = sum + LRModel.get(feature).get(i);

        }

        return sum / (recordCount * threshold);

    }

    public double findSigma(int feature, double mean, double threshold) {

        double summation = 0;

        for (int i = 0; i < recordCount * threshold; i++) {
            summation = summation + Math.pow(LRModel.get(feature).get(i) - mean, 2);
        }

        summation = summation / (recordCount * threshold);

        return Math.sqrt(summation);
    }


    public void transposeModel(double threshold) {

        ArrayList<ArrayList<Double>> transposedLR = new ArrayList<>();

        if (LRModel.size() == featureCount) { // Model is in normalization configuration, change to convergence configuration

            transposeElements(transposedLR, recordCount * threshold, featureCount);

        } else { // Model is in convergence configuration, change to normalization configuration

            transposeElements(transposedLR, featureCount, recordCount * threshold);

        }
        LRModel = transposedLR;
    }

    public void transposeElements(ArrayList<ArrayList<Double>> transposedLR, double v, double v2) {

        for (int i = 0; i < v; i++) {
            transposedLR.add(new ArrayList<>());
            for (int j = 0; j < v2; j++) {
                transposedLR.get(i).add(LRModel.get(j).get(i));
            }
        }
    }

    public void graphMaker() {

       try {
           FileWriter fw = new FileWriter("graph.csv");
           for(double z: losses){
               fw.write(losses+"\n");
           }

       }catch (Exception ex){
           ex.printStackTrace();
       }

    }

    public void batchGradientDescent(double learningRate, double error, double threshold) {

        System.out.print("\n\nStarting Gradient Descent:\n");
        System.out.print("--------------------------------------------------------------\n\n");


        transposeModel(threshold);

        for (int i = 0; i < featureCount; i++) {
            modelWeights.add(0.0);
        }

        double newLoss = 1;
        int epochCount = 0;

        while (convergenceCriteria(newLoss, error)) {

            double cost;
            lossValue = newLoss;
            ArrayList<Double> weightsCollection = new ArrayList<>();

            for (int k = 0; k < featureCount; k++) {
                weightsCollection.add(modelWeights.get(k) - (learningRate * partialDerivative(k, threshold)));
            }

            newLoss = lossFunction(threshold);
            modelWeights = weightsCollection;
            epochCount++;
            cost = Math.abs(lossValue - newLoss) * 100 / lossValue;

            losses.add(lossValue);

            if (epochCount == 1) {
                System.out.printf("Epoch %-4d%-5s Loss of %-10.3f Delta = %-10.5s \t Epsilon =  %-10.3s\n", epochCount, ":", lossValue,
                        "N/A", "N/A");

            } else {
                System.out.printf("Epoch %-4d%-5s Loss of %-10.3f Delta = %.5f%% \t Epsilon =  %.3f%%\n", epochCount, ":", lossValue,
                        cost, error);
            }


        }


        System.out.println("\nEpochs required:\t" + epochCount + "\n");

        System.out.println("Resulting weights:");
        for (int i = 0; i < featureCount; i++) {
            if (i == 0) {
                System.out.printf("W%-3d%s%10.3f\t (y-intercept)\n", i, ":", modelWeights.get(i));

            } else {
                System.out.printf("W%-3d%s%10.3f\n", i, ":", modelWeights.get(i));

            }
        }

    }

    public double partialDerivative(int observation, double threshold) {

        double sum = 0;

        for (int i = 0; i < recordCount * threshold; i++) {
            sum = sum + ((targetFeature.get(i) - hypothesis(i)) * -(LRModel.get(i).get(observation)));
        }


        return sum / (recordCount * threshold);
    }

    public boolean convergenceCriteria(double newLoss, double error) {

        double costNum = Math.abs(lossValue - newLoss) * 100;
        double percentCost = costNum / lossValue;

        return percentCost > error;
    }

    public double lossFunction(double threshold) {

        double summation = 0;

        for (int j = 0; j < recordCount * threshold; j++) {
            summation = summation + Math.pow((targetFeature.get(j) - hypothesis(j)), 2);
        }

        return (summation / (2 * (recordCount * threshold)));
    }

    public double hypothesis(int observation) {

        double sum = 0;

        for (int i = 0; i < featureCount; i++) {
            sum = sum + (LRModel.get(observation).get(i) * modelWeights.get(i));
        }

        return sum;
    }
}
