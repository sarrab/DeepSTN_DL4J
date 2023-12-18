package com.deepstn.training;

import com.deepstn.model.DeepSTN;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

public class DSTNPlusTrainer {
    private static final Logger log = LoggerFactory.getLogger(DSTNPlusTrainer.class);

    private final Object[] allData;
    private final Properties config;


    public DSTNPlusTrainer(Object[] allData, Properties config) {
        this.allData = allData;
        this.config = config;


    }

    public void train() {

        int iterateNum = Integer.parseInt(config.getProperty("iterateNum"));
        int batchSize = Integer.parseInt(config.getProperty("batchSize"));
        int epochs = Integer.parseInt(config.getProperty("epochs"));
        double lr = Double.parseDouble(config.getProperty("learning_rate"));
        double drop = Double.parseDouble(config.getProperty("drop"));
        long seed = Long.parseLong(config.getProperty("seed"));

        int channel = Integer.parseInt(config.getProperty("channels"));
        int height = Integer.parseInt(config.getProperty("grid_height"));
        int width = Integer.parseInt(config.getProperty("grid_width"));
        int residualUnits = Integer.parseInt(config.getProperty("nb_residual_unit"));
        int pre_F = Integer.parseInt(config.getProperty("nb_pre_filter"));
        int conv_F = Integer.parseInt(config.getProperty("nb_conv_filter"));
        int pooling_rate = Integer.parseInt(config.getProperty("pooling_rate"));
        int plusFilters = Integer.parseInt(config.getProperty("plus_filters"));


        int lenCloseness = Integer.parseInt(config.getProperty("len_closeness"));
        int lenPeriod = Integer.parseInt(config.getProperty("len_period"));
        int lenTrend = Integer.parseInt(config.getProperty("len_trend"));



        INDArray xTrain = (INDArray) allData[0];
        INDArray yTrain = (INDArray) allData[2];
        INDArray xTest = (INDArray) allData[3];
        INDArray yTest = (INDArray) allData[5];


        double[][] RMSE = new double[iterateNum][1];
        double[][] MAE = new double[iterateNum][1];
        boolean is_plus = true;
        boolean is_plus_efficient = false;
        boolean is_pt = false;

        boolean isPT_F = false;
        int P_N = 0;
        int T_F = 0;
        int PT_F = 0;
        int T = 0;
        int kernel_size_early_fusion = 1;
        log.info("**************************** conv model *******************************");
        ComputationGraph model = new DeepSTN().buildModel(height, width, channel, lenCloseness, lenPeriod, lenTrend, pre_F, conv_F, residualUnits, is_plus, is_plus_efficient,
                plusFilters, pooling_rate, is_pt, P_N, T_F, PT_F, T, drop, lr, kernel_size_early_fusion, isPT_F, seed);


        DataSet dataSet = new DataSet(xTrain, yTrain);
        DataSetIterator dataSetIterator = new ListDataSetIterator<>(dataSet.asList(), batchSize);



        long startTime = System.currentTimeMillis();


        log.info("************************ Training  ************************");
        model.fit(dataSetIterator, epochs);
        RegressionEvaluation eval = new RegressionEvaluation();

        INDArray[] output = model.output(false, xTest);

        eval.eval(output[0], yTest);
        log.info(eval.stats());


        long endTime = System.currentTimeMillis();

        log.info("cost time : " + (endTime - startTime));


    }


}
