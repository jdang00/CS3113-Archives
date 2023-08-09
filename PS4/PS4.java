/********************************
 Name:          Justin Dang
 Username:      ua711
 Problem Set:   PS4
 Due Date:      13 April 2023
 ********************************/
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PS4 {

    double[][] X = new double[10000][785];
    double[][] Y = new double[10000][10];
    int[] YRaw = new int[10000];
    double[][] YHat = new double[10000][10];

    double[][] YHatEncoded = new double[10000][10];

    double[][] W1 = new double[30][785];
    double[][] W2 = new double[10][31];

    double[][] H = new double[10000][30];

    double[][] delta1 = new double[10000][30];
    double[][] delta2 = new double[10000][10];

    double[][] grad1 = new double[30][785];
    double[][] grad2 = new double[10][30];

    double eta = 0.25;
    double lambda = 0.0001;
    double prevLoss = 0;

    public static void main(String[] args) {
        PS4 snn = new PS4();
        snn.readFile(args);
        snn.forwardProp();
        snn.trainNetwork();
        snn.test(args[3]);
    }

    public void printConsole(String[] args){
        System.out.print("""

                ************************************************************
                Problem Set:  Problem Set 4:  Neural Network
                Name:         Justin Dang
                Synax:        java PS4  w1.txt  w2.txt  xdata.txt  ydata.txt
                ************************************************************""");
        System.out.printf("\nTraining Phase:  "+args[2]+" \n" +
                "--------------------------------------------------------------\n" +
                "    => Number of Entries (n):              "+X.length+"\n" +
                "    => Number of Features (p):             " +(X[0].length - 1) +"\n");
    }
    public void test(String filename){

        YRaw = buildY(filename);
        buildYHat();

        System.out.print("""

                Testing Phase (first 10 records):
                --------------------------------------------------------------
                """);


        int right = 0;
        for(int i = 1; i < 11; i++){
            if(YRaw[i] == yHatDecode(YHatEncoded[i])){
                System.out.println("\tTest Record " + i + ":\t" + YRaw[i] +"\tPrediction: " + yHatDecode(YHatEncoded[i]) + "\tCorrect:\tTRUE");
                right++;
            }else{
                System.out.println("\tTest Record " + i + ":\t" + YRaw[i] +"\tPrediction: " + yHatDecode(YHatEncoded[i]) + "\tCorrect:\tFALSE");

            }
        }

        double percent = (right / 10.00) * 100;
        System.out.printf("\n\t=> Number of Test Entries (n): %10.0f", 10.0);
        System.out.printf("\n\t=> Accuracy: %.2f%%", percent);

    }

    public int yHatDecode(double[] vec){
        for(int i = 0; i < vec.length; i++){
            if(vec[i] == 1){
                return i + 1;
            }
        }
        return 0;
    }
    public void buildYHat(){
        for (int i = 0; i < YHat.length; i++) {
            YHatEncoded[i] = oneHotEncoding(maxValue(YHat[i]));
        }
    }
    public static double[][] multiply(double[][] A, double[][] B) {

        int rows1 = A.length;
        int cols1 = A[0].length;
        int rows2 = B.length;
        int cols2 = B[0].length;

        if (cols1 != rows2) {
            System.out.println(cols1 + " vs " + rows2);
            System.err.println("Error with multiplication!  Check the dimensions.");
            throw new IllegalArgumentException();
        }

        double[][] C = new double[rows1][cols2];
        for (int i = 0; i < rows1; i++) {
            Arrays.fill(C[i], 0.00000);
        }

        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < cols2; j++) {
                for (int k = 0; k < cols1; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }
    public static double[][] transpose(double[][] m) {
        double[][] temp = new double[m[0].length][m.length];

        for (int i = 0; i < m[0].length; i++)
            for (int j = 0; j < m.length; j++)
                temp[i][j] = m[j][i];

        return temp;
    }
    public static double[][] dropFirstColumn(double[][] m) {
        double[][] temp = new double[m.length][m[0].length - 1];

        for (int i = 0; i < temp.length; i++) {
            System.arraycopy(m[i], 1, temp[i], 0, temp[i].length);
        }
        return temp;
    }
    public void trainNetwork() {
        System.out.print("""

                Starting Gradient Descent:
                --------------------------------------------------------------
                """);

        for (int i = 1; i < 701; i++) {
            forwardProp();
            backProp();

            if(i == 1){
                System.out.printf("\nEpoch %-4d%-5s Loss of %-10.3f Delta = %-10.5s", i, ":", loss(),
                        "N/A");
            }else{
                System.out.printf("\nEpoch %-4d%-5s Loss of %-10.3f Delta = %.0f%%", i, ":", loss(),
                        calcDelta(prevLoss, loss()) * 100);
            }

            writeFile(i, String.format("%.3f",loss()), "loss.txt");
            writeFile(i, String.format("%.3f",accuracy()), "accuracy.txt");
            prevLoss = loss();

        }

        System.out.println("""


                Epochs Required:\t700""");


    }
    public double calcDelta(double newLoss, double loss){
        return Math.abs(loss - newLoss) * 100 / loss;
    }

    public double loss() {
        double loss = 0;
        for (int i = 0; i < YHat.length; i++) {

            for (int j = 0; j < YHat[0].length; j++) {
                loss = loss + (-Y[i][j] * Math.log(YHat[i][j]) - (1 - Y[i][j]) * Math.log(1 - YHat[i][j]));
            }
        }
        return ((double) 1 / YHat.length) * (loss + calcRegularization());
    }
    public double accuracy() {

        for (int i = 0; i < YHat.length; i++) {
            YHatEncoded[i] = oneHotEncoding(maxValue(YHat[i]));
        }
        int correct = 0;

        for (int i = 0; i < YHat.length; i++) {
            for (int j = 0; j < YHat[0].length; j++) {
                if (YHatEncoded[i][j] == Y[i][j] && YHatEncoded[i][j] != 0) {
                    correct++;
                }
            }

        }
        return (double) correct / X.length;

    }
    public void backProp() {
        deltaTwo();
        deltaOne();
        grad2calc();
        grad1calc();
    }
    public void deltaTwo() {
        for (int i = 0; i < delta2.length; i++) {
            for (int j = 0; j < delta2[0].length; j++) {
                delta2[i][j] = YHat[i][j] - Y[i][j];
            }
        }
    }
    public void deltaOne() {

        double[][] D2W2 = multiply(delta2, dropFirstColumn(W2));
        double[][] XW1T = multiply(dropFirstColumn(X), transpose(dropFirstColumn(W1)));

        for (int i = 0; i < XW1T.length; i++) {
            for (int j = 0; j < XW1T[0].length; j++) {
                XW1T[i][j] = activate(XW1T[i][j]) + (1 - activate(XW1T[i][j]));
            }
        }

        for (int i = 0; i < delta1.length; i++) {
            for (int j = 0; j < delta1[0].length; j++) {
                delta1[i][j] = D2W2[i][j] * XW1T[i][j];
            }
        }

    }
    public void grad2calc() {
        double[][] gradBias = new double[grad2.length][grad2[0].length + 1];
        double[][] grad2reg = new double[grad2.length][grad2[0].length];

        for (int i = 0; i < grad2.length; i++) {
            gradBias[i][0] = 0;
            System.arraycopy(grad2[i], 0, gradBias[i], 1, grad2[0].length);
        }

        gradBias = multiply(transpose(delta2), H);

        for (int i = 0; i < gradBias.length; i++) {
            for (int j = 0; j < gradBias[0].length; j++) {
                grad2reg[i][j] = (gradBias[i][j] / X.length)  + (weightsSquared(dropFirstColumn(W2))*(lambda/X.length));
            }
        }

        for(int i = 0; i < grad2.length; i++){
            for(int j = 0; j < grad2[0].length; j++){
                W2[i][j] = W2[i][j] - (eta * grad2reg[i][j]);
            }
        }

    }
    public void grad1calc() {
        grad1 = multiply(transpose(delta1), X);
        double[][] grad1reg = new double[grad1.length][grad1[0].length];

        for (int i = 0; i < grad1reg.length; i++) {
            for (int j = 0; j < grad1reg[0].length; j++) {
                grad1reg[i][j] = (grad1[i][j] / X.length) + (weightsSquared(dropFirstColumn(W1))*(lambda/X.length));
            }
        }

        for(int i = 0; i < grad1reg.length; i++) {
            for (int j = 0; j < grad1reg[0].length; j++) {
                W1[i][j] = W1[i][j] - (eta * grad1reg[i][j]);
            }
        }

    }
    public double calcRegularization() {
        double W2star = weightsSquared(W2);
        double W1star = weightsSquared(W1);
        return (W2star + W1star) * (lambda / (2 * YHat.length));
    }
    public double weightsSquared(double[][] weight) {
        double sum = 0;
        for (double[] doubles : weight) {
            for (int j = 0; j < weight[0].length; j++) {
                sum = sum + Math.pow(doubles[j], 2);
            }
        }
        return sum;
    }
    public void forwardProp() {

        H = multiply(X, transpose(W1));

        double[][] hBias = new double[10000][31];
        for (int i = 0; i < H.length; i++) {
            hBias[i][0] = 1;
            for (int j = 0; j < H[0].length; j++) {
                hBias[i][j + 1] = activate(H[i][j]);
                H[i][j] = activate(H[i][j]);
            }
        }

        YHat = multiply(hBias, transpose(W2));


        for (int i = 0; i < YHat.length; i++) {
            for (int j = 0; j < YHat[0].length; j++) {
                YHat[i][j] = activate(YHat[i][j]);
            }
        }

    }
    public double activate(double term) {

        return (1 / (1 + Math.exp(-term)));

    }
    public void printArrays(double[][] m) {
        for (double[] doubles : m) {
            for (int j = 0; j < m[0].length; j++) {
                System.out.print(doubles[j] + ", ");
            }
            System.out.println();
        }

    }
    public void readFile(String[] args) {
        createFile();
        printConsole(args);
        for (int i = 0; i < args.length; i++) {
            try {
                BufferedReader br = new BufferedReader(new FileReader(args[i]));

                String line;

                int rowCounter = 0;
                while ((line = br.readLine()) != null) {

                    String[] lineData = line.split(",");

                    double[] doubleValues = Arrays.stream(lineData)
                            .mapToDouble(Double::parseDouble)
                            .toArray();

                    loadVector(doubleValues, rowCounter, i);
                    rowCounter++;

                }

                br.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }

    }
    public void loadVector(double[] vector, int row, int target) {

        if (target == 0) {
            System.arraycopy(vector, 0, W1[row], 0, vector.length);
        } else if (target == 1) {
            System.arraycopy(vector, 0, W2[row], 0, vector.length);
        } else if (target == 2) {

            for (int i = 0; i < row; i++) {
                X[row][0] = 1.00;
                System.arraycopy(vector, 0, X[row], 1, vector.length);
            }

        } else if (target == 3) {
            double[] vectorEncode = oneHotEncoding((int) (vector[0]));
            System.arraycopy(vectorEncode, 0, Y[row], 0, vectorEncode.length);
        } else {
            System.out.println("Not able to find target array.");
            System.exit(100);
        }

    }
    public double[] oneHotEncoding(int value) {

        double[] encodedVector = new double[W2.length];

        if (value == 0) {

            for (int i = 0; i < W2.length - 2; i++) {
                encodedVector[i] = 0;
            }
            encodedVector[W2.length - 1] = 1;

        } else {
            for (int i = 0; i < W2.length; i++) {

                if (i == value - 1) {
                    encodedVector[i] = 1;
                } else {
                    encodedVector[i] = 0;
                }
            }
        }

        return encodedVector;

    }
    public int[] buildY(String filename){
        List<Integer> yVals = new ArrayList<>();

        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));

            String line;

            while ((line = br.readLine()) != null) {
                yVals.add(Integer.parseInt(line));
            }

            br.close();
        } catch(Exception ex){
            ex.printStackTrace();
        }

        return yVals.stream().mapToInt(i -> i).toArray();
    }
    public int maxValue(double[] array) {

        double max = array[0];
        int encoder = 0;

        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                encoder = i;
            }
        }

        return encoder + 1;
    }

    public void writeFile(int epoch, String num, String file){

        File log = new File(file);

        try{

            FileWriter fileWriter = new FileWriter(log, true);

            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            bufferedWriter.write("Epoch " + epoch + " " + num + "\n");
            bufferedWriter.close();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    public void createFile() {

        try {
            File loss = new File("loss.txt");
            File accuracy = new File("accuracy.txt");
            if (loss.createNewFile() && accuracy.createNewFile()) {

            } else {
                new FileWriter("loss.txt").close();
                new FileWriter("accuracy.txt").close();

            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

    }

}
