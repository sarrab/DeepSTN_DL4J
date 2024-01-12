package com.deepstn.training;

import com.deepstn.model.DeepSTN;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.Properties;

public class DSTNPlusTrainer {
    private static final Logger log = LoggerFactory.getLogger(DSTNPlusTrainer.class);

    private final Object[] allData;
    private final Properties config;


    public DSTNPlusTrainer(Object[] allData, Properties config) {
        this.allData = allData;
        this.config = config;

    }

    @SuppressWarnings("GrazieInspection")
    public void train() {

        //Configuration parameters
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


        // Flag to indicate whether or not we  include PoI*Time
        boolean isIncludePoiTime = Boolean.parseBoolean(config.getProperty("is_pt"));

        //Flag to indicate whether or not we add an aditional convolution layer to poi_time component
        boolean isPTmoreConv = Boolean.parseBoolean(config.getProperty("isPT_F"));
        int poiNum = Integer.parseInt(config.getProperty("P_N"));
        int numTimeFeatures = Integer.parseInt(config.getProperty("T_F"));
        int numPoiTimeFeatures = Integer.parseInt(config.getProperty("PT_F"));
        int timeInterval = Integer.parseInt(config.getProperty("T"));
        int kernel_size_early_fusion = Integer.parseInt(config.getProperty("kernel_size_early_fusion"));

        int saveModelInterval = Integer.parseInt(config.getProperty("saveModelInterval"));

        INDArray[] trainInput;
        INDArray[] trainLabel;
        INDArray[] testInput;
        INDArray[] testLabel;


        if (isIncludePoiTime) {

            INDArray xTrain = (INDArray) allData[0];
            INDArray tTrain = (INDArray) allData[1];
            INDArray poiTrain = (INDArray) allData[2];
            INDArray yTrain = (INDArray) allData[3];

            INDArray xTest = (INDArray) allData[4];
            INDArray tTest = (INDArray) allData[5];
            INDArray poiTest = (INDArray) allData[6];
            INDArray yTest = (INDArray) allData[7];

            trainInput = new INDArray[]{xTrain, poiTrain, tTrain};
            trainLabel = new INDArray[]{yTrain};
            testInput = new INDArray[]{xTest, poiTest, tTest};
            testLabel = new INDArray[]{yTest};
        } else {
            INDArray xTrain = (INDArray) allData[0];
            INDArray yTrain = (INDArray) allData[2];
            INDArray xTest = (INDArray) allData[3];
            INDArray yTest = (INDArray) allData[5];

            trainInput = new INDArray[]{xTrain};
            trainLabel = new INDArray[]{yTrain};
            testInput = new INDArray[]{xTest};
            testLabel = new INDArray[]{yTest};
        }

        MultiDataSet trainDataSet = new MultiDataSet(trainInput, trainLabel);
        MultiDataSet testDataSet = new MultiDataSet(testInput, testLabel);

        Iterator<org.nd4j.linalg.dataset.api.MultiDataSet> trainIterator = trainDataSet.asList().iterator();
        Iterator<org.nd4j.linalg.dataset.api.MultiDataSet> testIterator = testDataSet.asList().iterator();

        IteratorMultiDataSetIterator trainDataSetIterator = new IteratorMultiDataSetIterator(trainIterator, batchSize);
        IteratorMultiDataSetIterator testDataSetIterator = new IteratorMultiDataSetIterator(testIterator, batchSize);


        log.info("****************** Building the model *********************");
        ComputationGraph model = new DeepSTN().buildModel(height, width, channel, lenCloseness, lenPeriod, lenTrend, pre_F, conv_F, residualUnits,
                plusFilters, pooling_rate, isIncludePoiTime, poiNum, numTimeFeatures, numPoiTimeFeatures, timeInterval, drop, lr, kernel_size_early_fusion, isPTmoreConv, seed);


        long startTime = System.currentTimeMillis();


        log.info("******************* Training  ************************");


        for (int epoch = 0; epoch < epochs; epoch++) {
            model.fit(trainDataSetIterator);

            // Model saving and evaluation every n epochs or at the last epoch

            if ((epoch + 1) % saveModelInterval == 0 || epoch == epochs - 1) {
                String modelFilename = "model_at_epoch_" + (epoch + 1) + ".zip";
                try {
                    ModelSerializer.writeModel(model, modelFilename, true);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                log.info("Saved model to " + modelFilename);

                // Evaluate the model
                RegressionEvaluation eval = model.evaluateRegression(testDataSetIterator);
                log.info("Evaluation at epoch " + (epoch + 1) + ": " + eval.stats());
                String statsFilename = "stats_epoch_" + (epoch + 1) + ".txt";
                try (BufferedWriter writer = new BufferedWriter(new FileWriter(statsFilename))) {
                    writer.write(eval.stats());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        long endTime = System.currentTimeMillis();
        log.info("Training completed in " + (endTime - startTime) + " ms");
    }




}
