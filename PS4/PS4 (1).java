/********************************
Name: Ivan Welborn
Problem Set: PS4
Due Date: April 13, 2023
********************************/

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

public class PS4 {

	double[][] w1 = new double[30][785];
	double[][] w2 = new double[10][31];
	double[][] x = new double[10000][785];
	int[][] y = new int[10000][10];
	int[] actYs = new int[10000];
	double[][] h1 = new double[10000][30];
	double[][] yHat = new double[10000][10];
	int[][] yEncoded = new int[10000][10];
	int[] yHatAct = new int[10000];
	double[][] delta2 = new double[10000][10];
	double[][] delta1 = new double[10000][30];
	double[][] nabby2 = new double[10][30];
	double[][] nabby1 = new double[30][785];
	double alpha = 0.25;
	File loss = new File("loss.txt");
	File accuracy = new File("accuracy.txt");
	public static void main(String[] args) {

		if (args.length == 4) {
			BufferedWriter bw;
			PS4 p = new PS4();
			p.readEm(args[0], args[1], args[2], args[3]);
			p.printIt(args[0], args[1], args[2], args[3]);
			p.gradientDescent();
			try {
				bw = new BufferedWriter(new FileWriter("w1out.txt", true));
				for(int i = 0; i < p.w1.length; i++) {
					for(int j = 0; j < p.w1[i].length; j++) {
						bw.append(p.w1[i][j] + ",");
					}
					bw.append("\n");
				}
				bw.close();
				bw = new BufferedWriter(new FileWriter("w2out.txt", true));
				for(int i = 0; i < p.w2.length; i++) {
					for(int j = 0; j < p.w2[i].length; j++) {
						bw.append(p.w2[i][j] + ",");
					}
					bw.append("\n");
				}
				bw.close();
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		} else {
			System.out.println("Invalid amount of arguments");
		}
	}

	public void printIt(String a, String b, String c, String d) {
		String[] w1 = a.split("\\\\");
		String[] w2 = b.split("\\\\");
		String[] xData = c.split("\\\\");
		String[] yData = d.split("\\\\");
		System.out.printf("************************************************************\n" + "Problem Set:%20s%20s\n"
				+ "Name:%25s\n", "Problem Set 4:", "Nueral Network", "Ivan Welborn");
		System.out.printf("Synax:%21s%s%s%s%s\n************************************************************\r\n" + "",
				"java PS4 ", w1[w1.length - 1] + " ", w2[w2.length - 1] + " ", xData[xData.length - 1] + " ",
				yData[yData.length - 1] + " ");
		System.out.printf(
				"\nTraining Phase:\t%s\n--------------------------------------------------------------\r\n"
						+ "\t=> Number of Entries (n):%20d\n" + "\t=> Number of Entries (p):%20d\n"
						+ "\n\nStarting Gradient Descent:\n"
						+ "--------------------------------------------------------------\r\n\n",
				c, x.length, x[0].length - 1);
	}

	public void gradientDescent() {
		int epochs = 0;
		while (epochs < 700) {
			double oldLoss = loss();
			forwardProp();
			backwardProp();
			epochs = printEpochUpdate(epochs, oldLoss);
			try {
				BufferedWriter bw = new BufferedWriter(new FileWriter(loss, true));
				bw.write(String.format("Epoch: " + epochs + "\tLoss: %.3f\n" ,loss()));
				bw.close();
				bw = new BufferedWriter(new FileWriter(accuracy, true));
				bw.append(String.format("Epoch: " + epochs + "\tAccuracy: %.3f\n" ,(predictions()/10000) * 100));
				bw.close();
			}catch(Exception ex) {
				ex.printStackTrace();
			}
		}
		printEnding(epochs);
	}

	public void forwardProp() {
		h1 = findH(x, w1);
		double[][] h1Biased = addBias(h1);
		yHat = findH(h1Biased, w2);
		yEncoded = buildEncodedYs();
	}

	public void backwardProp() {
		buildDelta2();
		buildDelta1();
		nabby2 = nabla2();
		nabby1 = nabla1();
		adjustWeights();
	}

	public int printEpochUpdate(int epochs, double oldLoss) {
		epochs++;
		double numerator = oldLoss - loss();
		if (epochs == 1) {
			System.out.printf("Epoch: %d\tLoss of %.2f\tN/A %-15sN/A", epochs, loss(), "");
			// System.out.printf("Accuracy: %.3f \n", (predictions() / 10000) * 100);
		} else {
			double epsilon = (predictions() / 10000);
			// epsilon = 1 - epsilon;
			System.out.printf("Epoch: %d\tLoss of %.2f\tDelta = %.2f%% %-4sEpsilon = %.1f%%", epochs, loss(),
					(numerator / oldLoss) * 100, "", epsilon * 100);
		}
		System.out.println();
		return epochs;
	}

	public void printEnding(int epochs) {

		System.out.printf("\nEpochs Required:\t%d", epochs);
		System.out.println("\n\nTesting Phase (first 10 records):\n"
				+ "--------------------------------------------------------------\r\n");
		// double howMany = 0;

		for (int i = 0; i < 10; i++) {
			forwardProp();
			System.out.printf("\tTest Record %d:%3d\tPrediction: %d\tCorrect:\t", i + 1, actYs[i], yHatAct[i]);
			if (actYs[i] == yHatAct[i]) {
				System.out.println("TRUE");
			} else {
				System.out.println("FALSE");
			}
		}
		System.out.printf("\n\t=> Number of Test Entries (n): %15d\n\t=>Accuracy: %37.2f%%", 10,
				(predictions() / 10000) * 100);
	}

	public double loss() {
		double sum = 0;
		for (int i = 0; i < y.length; i++) {
			for (int j = 0; j < y[i].length; j++) {
				sum += (Math.negateExact(y[i][j]) * Math.log(yHat[i][j])) - ((1 - y[i][j]) * Math.log(1 - yHat[i][j]));
			}
		}
		sum /= y.length;
		sum += l2RegDivided(w1, w2);
		return sum;

	}

	public double predictions() {
		int in = 0;
		for (int i = 0; i < yEncoded.length; i++) {
			if (actYs[i] == yHatAct[i]) {
				in++;
			}
		}
		return in;
	}

	public void adjustWeights() {
		double reg = l2Reg(w1) * (0.0001 / 10000);
		double[][] nabby1Copy = new double[w1.length][w1[0].length];

		for (int i = 0; i < nabby1.length; i++) {
			for (int j = 0; j < nabby1[i].length; j++) {
				nabby1Copy[i][j] = (nabby1[i][j] / 10000) + reg;
			}
		}

		for (int i = 0; i < w1.length; i++) {
			for (int j = 0; j < w1[i].length; j++) {
				w1[i][j] = w1[i][j] - (alpha * nabby1Copy[i][j]);
			}
		}

		reg = l2Reg(w2) * (0.0001 / 10000);
		double[][] nabby2Copy = new double[nabby2.length][nabby2[0].length + 1];
		double[][] nabby2Play = new double[w2.length][w2[0].length];
		for (int i = 0; i < nabby2Copy.length; i++) {
			for (int j = 0; j < nabby2Copy[i].length; j++) {
				if (j == 0) {
					nabby2Copy[i][j] = 0;
				} else {
					nabby2Copy[i][j] = nabby2[i][j - 1];
				}
			}
		}
		for (int i = 0; i < nabby2Copy.length; i++) {
			for (int j = 0; j < nabby2Copy[i].length; j++) {
				nabby2Play[i][j] = (nabby2Copy[i][j] / 10000) + reg;
			}
		}
		for (int i = 0; i < w2.length; i++) {
			for (int j = 0; j < w2[i].length; j++) {
				w2[i][j] = w2[i][j] - (alpha * nabby2Play[i][j]);
			}
		}
	}

	public double l2RegDivided(double[][] w1, double[][] w2) {
		double sum = 0;
		double[][] a = dropFirstColumn(w1);
		double[][] b = dropFirstColumn(w2);
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				sum += Math.pow(a[i][j], 2);
			}
		}
		for (int i = 0; i < b.length; i++) {
			for (int j = 0; j < b[i].length; j++) {
				sum += Math.pow(b[i][j], 2);
			}
		}
		sum *= (0.0001 / (2 * 10000));
		return sum;
	}

	public double l2Reg(double[][] w) {
		double sum = 0;
		double[][] a = dropFirstColumn(w);
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				sum += Math.pow(a[i][j], 2);
			}
		}
		return sum;
	}

	public double[][] nabla1() {
		double[][] re = multiply(transpose(delta1), x);
		return re;
	}

	public double[][] nabla2() {
		double[][] re = multiply(transpose(delta2), h1);
		return re;
	}

	public void buildDelta1() {
		double[][] left = multiply(delta2, dropFirstColumn(w2));
		double[][] right = multiply(dropFirstColumn(x), transpose(dropFirstColumn(w1)));
		for (int i = 0; i < right.length; i++) {
			for (int j = 0; j < right[i].length; j++) {
				right[i][j] = activation(right[i][j]) * (1 - activation(right[i][j]));
			}
		}
		delta1 = hallmarkProduct(left, right);
	}

	public double[][] hallmarkProduct(double[][] a, double[][] b) {
		double[][] re = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				re[i][j] = a[i][j] * b[i][j];
			}
		}
		return re;
	}

	public void buildDelta2() {
		for (int i = 0; i < delta2.length; i++) {
			for (int j = 0; j < delta2[0].length; j++) {
				delta2[i][j] = yHat[i][j] - y[i][j];
			}
		}
	}

	public int[][] buildEncodedYs() {
		int[][] re = new int[yEncoded.length][yEncoded[0].length];
		for (int row = 0; row < yEncoded.length; row++) {
			int tRow = findMax(row);
			for (int col = 0; col < yEncoded[row].length; col++) {
				if (col == tRow) {
					re[row][tRow] = 1;
					yHatAct[row] = col + 1;
				}
			}
		}
		return re;
	}

	public int findMax(int row) {
		double max = yHat[row][0];
		int theOne = 0;
		for (int i = 0; i < yHat[row].length; i++) {
			// This will change our decimal values into the one-hot encoded stuff
			if (max < yHat[row][i]) {
				max = yHat[row][i];
				theOne = i;
			}
		}
		return theOne;
	}

	public double[][] addBias(double[][] h) {
		double[][] output = new double[h.length][h[0].length + 1];
		for (int i = 0; i < h.length; i++) {
			output[i][0] = 1.0;
			for (int j = 0; j < h[i].length; j++) {
				output[i][j + 1] = h[i][j];
			}
		}
		return output;
	}

	public double activation(double x) {
		double re = (1 / (1 + Math.exp(-x)));
		return re;
	}

	public double rowSum(int row, int col, double[][] in) {
		double it = in[row][col];
		double sum = 0;
		for (int i = 0; i < in[row].length; i++) {
			sum += in[row][i];
		}
		it = (it / sum);
		return it;
	}

	public double[][] findH(double[][] x, double[][] w) {
		double[][] re = multiply(x, transpose(w));
		for (int i = 0; i < re.length; i++) {
			for (int j = 0; j < re[i].length; j++) {
				re[i][j] = activation(re[i][j]);
			}
		}
		return re;
	}

	public void readEm(String wOne, String wTwo, String xD, String yD) {
		File file = new File(wOne);
		BufferedReader br;
		FileReader f;
		try {
			if (file.exists()) {
				f = new FileReader(wOne);
				br = new BufferedReader(f);
				String line = "";
				int row = 0;
				while ((line = br.readLine()) != null) {
					String[] data = line.split(",");
					for (int col = 0; col < w1[row].length; col++) {
						w1[row][col] = Double.parseDouble(data[col]);
					}
					row++;
				}
				br.close();
			} else {
				System.out.println("Invalid file: " + wOne);
			}
			file = new File(wTwo);
			if (file.exists()) {
				f = new FileReader(wTwo);
				br = new BufferedReader(f);
				String line = "";
				int row = 0;
				while ((line = br.readLine()) != null) {
					String[] data = line.split(",");
					for (int col = 0; col < w2[row].length; col++) {
						w2[row][col] = Double.parseDouble(data[col]);
					}
					row++;
				}
				br.close();
			}
			file = new File(xD);
			if (file.exists()) {
				f = new FileReader(xD);
				br = new BufferedReader(f);
				String line = "";
				int row = 0;
				while ((line = br.readLine()) != null) {
					String[] data = line.split(",");
					x[row][0] = 1.0;
					for (int col = 0; col < data.length; col++) {
						x[row][col + 1] = Double.parseDouble(data[col]);
					}
					row++;
				}
				br.close();
			}
			file = new File(yD);
			if (file.exists()) {
				f = new FileReader(yD);
				br = new BufferedReader(f);
				String line = "";
				int row = 0;
				while ((line = br.readLine()) != null) {
					y[row][Integer.parseInt(line) - 1] = 1;
					actYs[row] = Integer.parseInt(line);
					row++;
				}
				br.close();
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	public static double[][] dropFirstColumn(double m[][]) {
		double[][] temp = new double[m.length][m[0].length - 1];

		for (int i = 0; i < temp.length; i++) {
			for (int k = 0; k < temp[i].length; k++) {
				temp[i][k] = m[i][k + 1];
			}
		}
		return temp;
	}

	public static double[][] transpose(double m[][]) {
		double[][] temp = new double[m[0].length][m.length];

		for (int i = 0; i < m[0].length; i++)
			for (int j = 0; j < m.length; j++)
				temp[i][j] = m[j][i];

		return temp;
	}

	public static double[][] multiply(double[][] A, double[][] B) {

		int rows1 = A.length;
		int cols1 = A[0].length;
		int rows2 = B.length;
		int cols2 = B[0].length;

		if (cols1 != rows2) {
			System.err.println("Error with multiplication!  Check the dimensions.");
			throw new IllegalArgumentException();
		}

		double[][] C = new double[rows1][cols2];
		for (int i = 0; i < rows1; i++) {
			for (int j = 0; j < cols2; j++) {
				C[i][j] = 0.00000;
			}
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

	public static void printDimensions(double[][] m) {
		String xdim = String.format(" Matrix dimensions:    %d x %d ", m.length, m[0].length);

		System.out.println(xdim);
	}

	public void print(double[][] input) {
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[i].length; j++) {
				System.out.printf("%.3f\t", input[i][j]);
			}
			System.out.print("\n");
		}
	}

	public void print(int[][] input) {
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[i].length; j++) {
				System.out.print(input[i][j] + " ");
			}
			System.out.print("\n");
		}
	}
}
