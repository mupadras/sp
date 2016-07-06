package work.spark.lr;

/**
 * Created by madhu on 7/5/16.
 */


import Jama.*;
import Jama.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;

import java.util.*;
import java.io.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;


public class simpleLinearRegression {


    public Double alpha;
    public String trainFile;
    public String testFile;

    public simpleLinearRegression() {
        alpha = 1.0;
        trainFile = "/Users/madhu/Desktop/regdata.txt";
        testFile = "/Users/madhu/Desktop/one.txt";
    }

    private Matrix CreateMatrix(String file) {

        try {

            //String filename = "/Users/madhu/Desktop/regdata.txt";
            FileReader fr = new FileReader(file);
            BufferedReader br = new BufferedReader(fr);
            List<double[]> createArray = new ArrayList<double[]>();

            //JavaSparkContext sc = new JavaSparkContext(conf);

            String line;
            String[] parts;
            //String[] features;
            while ((line = br.readLine()) != null) {
                parts = line.split(",");
                //features = line.split(",");
                String[] features = parts[1].split(" ");
                //JavaRDD<String> data = sc.textFile(filename);
                //JavaRDD<LabeledPoint> parsedData;
                //parsedData = DataMap(data);

                double[] v = new double[features.length];
                for (int i = 0; i < features.length; i++) {
                    v[i] = Double.parseDouble(features[i]);
                    //System.out.print(v[i] + "  ");
                    //createArray.add(data);
                    //new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
                    createArray.add(v);
                }
                //System.out.println("  ");

            }
            fr.close();

            if (createArray.size() > 0) {
                int column = createArray.get(0).length;
                int row = createArray.size();
                Matrix matrix = new Matrix(row, column);
                for (int r = 0; r < row; r++) {
                    for (int c = 0; c < column; c++) {
                        matrix.set(r, c, createArray.get(r)[c]);
                        return matrix;
                    }
                }

            }
            //return null;

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);

        }

        return new Matrix(0, 0);

    }

    private Matrix dataPointsObtain(Matrix dataGiven) {
        Matrix attributes = dataGiven.getMatrix(0, dataGiven.getRowDimension() - 1, 0, dataGiven.getColumnDimension() - 2);
        int row = attributes.getRowDimension();
        int column = attributes.getColumnDimension() + 1;
        Matrix modifiedAttributes = new Matrix(row, column);
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                if (c == 0) {
                    modifiedAttributes.set(r, c, 1.0);
                } else {
                    modifiedAttributes.set(r, c, attributes.get(r, c - 1));
                }
            }
        }
        return modifiedAttributes;
    }

    private Matrix obtainTarget(Matrix dataGiven) {
        return dataGiven.getMatrix(0, dataGiven.getRowDimension() - 1, dataGiven.getColumnDimension() - 1, dataGiven.getColumnDimension() - 1);
    }


    private Matrix trainModel(Matrix data, Matrix target, Double alpha) {
        int row = data.getRowDimension();
        int column = data.getColumnDimension();
        Matrix identity = Matrix.identity(column, column);
        identity.times(alpha);
        Matrix dataCopy = data.copy();
        Matrix transponseData = dataCopy.transpose();
        Matrix norm = transponseData.times(data);
        Matrix circular = norm.plus(identity);
        Matrix circularInverse = circular.inverse();
        Matrix former = circularInverse.times(data.transpose());
        Matrix giveWeight = former.times(target);

        return giveWeight;
    }

    private double testLinearRegressionModel(Matrix data, Matrix target, Matrix giveWeights) {
        double error = 0.0;
        int row = data.getRowDimension();
        int column = data.getColumnDimension();
        assert row == target.getRowDimension();
        assert column == giveWeights.getColumnDimension();

        Matrix predictTarget = predict(data, giveWeights);
        for (int i = 0; i < row; i++) {
            error = (target.get(i, 0) - predictTarget.get(i, 0)) * (target.get(i, 0) - predictTarget.get(i, 0));
        }

        return 0.5 * error;
    }


    private Matrix predict(Matrix data, Matrix giveWeights) {
        int row = data.getRowDimension();
        Matrix predictTarget = new Matrix(row, 1);
        for (int i = 0; i < row; i++) {
            double value = multiply(data.getMatrix(i, i, 0, data.getColumnDimension() - 1), giveWeights);
            //System.out.println(value);
            predictTarget.set(i, 0, value);
        }
        return predictTarget;
    }


    private Double multiply(Matrix data, Matrix giveWeights) {
        Double sum = 0.0;
        int column = data.getColumnDimension();
        for (int i = 0; i < column; i++) {
            sum += data.get(0, i) * giveWeights.get(i, 0);
        }
        return sum;
    }


    public static void main(String[] args) throws Exception{

        simpleLinearRegression rg = new simpleLinearRegression();

        try {
            Matrix train = rg.CreateMatrix(rg.trainFile);
            Matrix test = rg.CreateMatrix(rg.testFile);

            /** get the actual attributes, meanwhile add a N*1 column vector with value being all 1 as the first column of the attributes */
            Matrix trainData = rg.dataPointsObtain(train);
            Matrix testData = rg.dataPointsObtain(test);

            Matrix trainTarget = rg.obtainTarget(train);
            Matrix testTarget = rg.obtainTarget(test);

            // Train the model.
            Matrix giveWeights = rg.trainModel(trainData, trainTarget, rg.alpha);
            for (int i = 0; i < giveWeights.getRowDimension(); i++) {
                //System.out.println(giveWeights.get(i, 0));
            }

            //Test the model using given data

            double train_error = rg.testLinearRegressionModel(trainData, trainTarget, giveWeights);
            double test_error = rg.testLinearRegressionModel(testData, testTarget, giveWeights);

            System.out.println("Train error: " + train_error);
            System.out.println("Test error: " + test_error);

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }


    }

}













