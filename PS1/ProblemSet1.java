import java.io.BufferedReader;
import java.io.FileReader;

/************************************
Name:           Justin Dang
Problem Set:    Problem Set 1
Due Date:       27 Friday 2023
************************************/


public class ProblemSet1 {
	
	public static void main(String[] args) {
		
		String studentName = "Justin Dang";
		
		/***************************************************************
		 * DO NOT CHANGE ANYTHING ELSE BELOW THIS LINE IN THIS METHOD
		 ***************************************************************/
		
		/*
		 * Note:  If you use an IDE, you can set the command-line arguments for testing purposes through
		 * the IDE's runtime configuration. Simply ask if you do not know how to do this.
		 */
		
		if ( args.length < 2) {
			System.out.println("Invalid syntax.  Usage:  java ProblemSet1 filename colnumber");
			return;
		}
		
		try {
			
			ProblemSet1 ps = new ProblemSet1();
			
			String filename = args[0];
			int col = Integer.parseInt(args[1]);
			
			System.out.printf("[ Input arguments ] %n");
			System.out.printf("   Student Name ..................... %s %n", studentName);
			System.out.printf("   Input file ....................... %s %n", filename);
			System.out.printf("   Input column ..................... %d %n", col);
			
			System.out.println();
			
			
			/***************************************************************
			 * Calling step 1
			 ***************************************************************/
			
			System.out.println("[ Results ]");
			System.out.print("Step 1) Reading the file .......................... ");
			double[][] X = ps.readFile(filename);
			System.out.println("[done]");
			 
			int numColumns = (X != null && X[0] != null ? X[0].length : 0);
			System.out.printf("      Total number of records ..................... %d %n", (X != null ? X.length : 0) ) ;
			System.out.printf("      Total number of columns ..................... %d %n", numColumns ) ;
			System.out.println();
			
			
			/***************************************************************
			 * Calling step 2
			 ***************************************************************/
			System.out.printf("Step 2) Obtaining average for column [%d] .......... ", col);
			double average = ps.getAverage(ps.getColumnVector(X, col));
			System.out.println("[done]");
			System.out.printf("      Requested column [%d] average ................ %.3f %n", col, average  ) ;
			System.out.println();
			for ( int i = 0; i < numColumns; i++ ) {
				average = ps.getAverage(ps.getColumnVector(X, i));
				System.out.printf("      Column [%d] average .......................... %.3f %n", i, average  ) ;	
			}
			System.out.println();
			
			
			
			/***************************************************************
			 * Calling step 3
			 ***************************************************************/
			
			System.out.printf("Step 3) Obtaining output for run1(x%d) ............. ", col);
			double run1Output = ps.run1(ps.getColumnVector(X, col));
			System.out.println("[done]");
			
			System.out.printf("      Output from run1(x%d) ........................ %.3f %n", col, run1Output  ) ;
			System.out.println();
			
			
			
			
			/***************************************************************
			 * Calling step 4
			 ***************************************************************/
			System.out.printf("Step 4) Obtaining output for run2(x%d) ............. ", col);
			double run2Output = ps.run2(X);
			System.out.println("[done]");
			
			System.out.printf("      Output from run2(x%d) ........................ %.3f %n", col, run2Output  ) ;

			
			System.out.println();
			
			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		
		
	}
	
	
	/***********************************************************************
	 * CODE TO IMPLEMENT:  Implement the following methods
	 ***********************************************************************/
	
	
	/*
	 * Return a matrix for the file provided as a command-line argument
	 */
	public double[][] readFile(String filename) {

		int column = 0;
		int row = 0;

		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));

			String line;

			while ((line = br.readLine()) != null) {
				String[] numArray = line.split(",");
				column = numArray.length;
				row++;

			}

			br.close();
		} catch(Exception ex){
			ex.printStackTrace();
		}

		double[][] kArray = new double[row][column];

		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));

			String line;
			int rowCount = 0;
			while ((line = br.readLine()) != null) {
				String[] lineArray = line.split(",");
				for( int i = 0; i < lineArray.length; i++){
					kArray[rowCount][i] = Double.parseDouble(lineArray[i]);
				}
				rowCount++;
			}

			br.close();
		} catch(Exception ex){
			ex.printStackTrace();
		}


		return kArray;
	}
	
	/*
	 * Return an array that represents the k-th column vector
	 */
	public double[] getColumnVector(double[][] data, int k) {

		double[] vectorColumn = new double[data.length];

		for(int i = 0; i < data.length; i++){
			vectorColumn[i] = data[i][k];
		}
		return vectorColumn;
	}
	
	/*
	 * Return the average for the vector provided
	 */
	public double getAverage(double[] x) {

		double sum = 0;

		for(double y: x){
			sum = sum + y;
		}

		return sum / x.length;
	}
	
	
	/*
	 * Return the calculation as indicated in the problem set
	 */
	public double run1(double[] x) {

		double summation = 0;

		for(int i = 0; i < x.length; i++){
			summation = summation + Math.pow((x[i] - getAverage(x)), 2);
		}

		return summation;
	}
	
	/*
	 * Return the calculation as indicated in the problem set
	 */
	
	public double run2(double[][] x) {

		double sumCall = 0;

		for(int i = 0; i < x[0].length; i++){
			double[] y = getColumnVector(x,i);
				sumCall = sumCall + Math.sqrt(run1(y));
		}

		return sumCall;
	}

}