package com.deepstn.training;

import com.deepstn.model.DeepSTN;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Properties;

public class DSTNPlusTrainer {
    public void train(Object[] allData, Properties config) {

        int iterateNum = Integer.parseInt(config.getProperty("iterateNum"));
        int batchSize = Integer.parseInt(config.getProperty("batch_size"));
        int epochs = Integer.parseInt(config.getProperty("epochs"));
        double lr = Double.parseDouble(config.getProperty("learning rate"));
        double drop = Double.parseDouble(config.getProperty("drop"));
        long seed = Long.parseLong(config.getProperty("seed"));
        int channel = Integer.parseInt(config.getProperty("channels"));
        int height = Integer.parseInt(config.getProperty("grid_height"));
        int width = Integer.parseInt(config.getProperty("grid_width"));
        int residualUnits = Integer.parseInt(config.getProperty("residual_units"));
        int pre_F = Integer.parseInt(config.getProperty("nb_pre_filter"));
        int conv_F = Integer.parseInt(config.getProperty("nb_conv_filter"));
        int pooling_rate = Integer.parseInt(config.getProperty("pooling_rate"));
        int plusFilters = Integer.parseInt(config.getProperty("plus_filters"));


        int lenTest = Integer.parseInt(config.getProperty("len_test"));
        int lenCloseness = Integer.parseInt(config.getProperty("len_closeness"));
        int lenPeriod = Integer.parseInt(config.getProperty("len_period"));
        int lenTrend = Integer.parseInt(config.getProperty("len_trend"));
        int TCloseness = Integer.parseInt(config.getProperty("T_closeness"));
        int TPeriod = Integer.parseInt(config.getProperty("T_period"));
        int TTrend = Integer.parseInt(config.getProperty("T_trend"));


        INDArray xTrain = (INDArray) allData[0];
        INDArray timeTrain = (INDArray) allData[1];
        INDArray yTrain = (INDArray) allData[2];
        INDArray xTest = (INDArray) allData[3];
        INDArray timeTest = (INDArray) allData[4];
        INDArray yTest = (INDArray) allData[5];
        double maxMin = (Double) allData[6];

        double[][] RMSE = new double[iterateNum][1];
        double[][] MAE = new double[iterateNum][1];
        boolean is_plus = true;
        boolean is_plus_efficient = false;
        boolean is_pt = false;
        boolean is_summary = false;

        boolean isPT_F = false;
        int P_N = 0;
        int T_F = 0;
        int PT_F = 0;
        int T = 0;
        int kernel_size_early_fusion = 3;
        System.out.println("************ conv model ************");
        ComputationGraph model = new DeepSTN().buildModel(height, width, channel, lenCloseness, lenPeriod, lenTrend, pre_F, conv_F, residualUnits, is_plus, is_plus_efficient,
                plusFilters, pooling_rate, is_pt, P_N, T_F, PT_F, T, drop, is_summary, lr, kernel_size_early_fusion, isPT_F, seed);
        //String file_conv = current_directory + "/DeepSTN_10/MODEL/DeepSTN_10_model_" + iterate + ".hdf5";
        model.init();
        int count = 0;
        for (int i = 0; i < iterateNum; i++) {
            count++;
            long timeStart = System.currentTimeMillis() / 1000;
            System.out.println("==========");
            System.out.println("***** training conv_model *****");
            model.fit(new INDArray[]{xTrain}, new INDArray[]{yTrain});


        }
        System.out.println(xTrain.shapeInfoToString());
    }

}
