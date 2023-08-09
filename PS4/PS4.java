import java.awt.desktop.SystemSleepEvent;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;

public class PS4 {
	double[][] w1 = new double[30][785];
	double[][] w2 = new double[10][31];
	double[][] x = new double[10000][785];
	int[][] y = new int[10000][10];
	double[][] h1 = new double[10000][31];
	double[][] yHat = new double[10000][10];
	
	public static void main(String[] args) {

		if (args.length == 4) {
			PS4 p = new PS4();
			p.readEm(args[0], args[1], args[2], args[3]);
			p.forwardProp();

		} else {
			System.out.println("Invalid amount of arguments");
		}
	}

	public void print(double[][] input) {
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[i].length; j++) {
				System.out.print(input[i][j] + " ");
			}
			System.out.print("\n");
		}
	}

	public void forwardProp() {
		h1 = findH(x, w1);
		h1 = addBias(h1);
		yHat = findH(h1, w2);
		yHat = softMax(yHat);
		print(yHat);
//		for(int i = 0; i < yHat.length; i++) {
//			System.out.println(findMax(i));
//		}
	}
	
	public int findMax(int row) {
		double max = yHat[row][0];
		int theOne = 0;
		for(int i = 0; i < yHat[row].length; i++) {
			if(max < yHat[row][i]) {
				max = yHat[row][i];
				theOne = i+1;
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
		h1 = output;
		return h1;
	}

	public double activation(double x) {
		double re = (1 / (1 + Math.exp(-x)));
		return re;
	}

	public double rowSum(int row, int col, double[][] in) {
		double it = in[row][col];
		double sum = 0;
		for(int i = 0; i < in[row].length; i++) {
				sum += in[row][i];
		}
		it = (it/sum);
		return it;
	}
	
	public double[][] softMax(double[][] input) {
		double[][] softMax = new double[input.length][input[0].length];
		for(int i = 0; i < softMax.length; i++) {
			for(int j = 0; j < softMax[i].length; j++) {
				softMax[i][j] = rowSum(i, j, input);
			}
		}
		return softMax;
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
				while ((line = br.readLine()) != null) {
					for (int i = 0; i < y.length; i++) {
						y[i][Integer.parseInt(line) - 1] = 1;
					}
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

}
