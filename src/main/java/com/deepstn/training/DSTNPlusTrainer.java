package com.deepstn.training;

import com.deepstn.model.DeepSTN;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
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


        boolean is_pt = Boolean.parseBoolean(config.getProperty("is_pt"));
        boolean isPT_F = Boolean.parseBoolean(config.getProperty("isPT_F"));
        int P_N = Integer.parseInt(config.getProperty("P_N"));
        int T_F = Integer.parseInt(config.getProperty("T_F"));
        int PT_F = Integer.parseInt(config.getProperty("PT_F"));
        int T = Integer.parseInt(config.getProperty("T"));

        int kernel_size_early_fusion = Integer.parseInt(config.getProperty("kernel_size_early_fusion"));

        log.info("**************************** conv model *******************************");
        ComputationGraph model = new DeepSTN().buildModel(height, width, channel, lenCloseness, lenPeriod, lenTrend, pre_F, conv_F, residualUnits,
                plusFilters, pooling_rate, is_pt, P_N, T_F, PT_F, T, drop, lr, kernel_size_early_fusion, isPT_F, seed);


        DataSet dataSet = new DataSet(xTrain, yTrain);
        DataSetIterator dataSetIterator = new ListDataSetIterator<>(dataSet.asList(), batchSize);
        DataSetIterator testDataSetIterator = new ListDataSetIterator<>(new DataSet(xTest, yTest).asList(), batchSize);


        long startTime = System.currentTimeMillis();


        log.info("************************ Training  ************************");


        for (int epoch = 0; epoch < epochs; epoch++) {
            model.fit(dataSetIterator);

            if ((epoch + 1) % 5 == 0 || epoch == epochs - 1) {
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

            dataSetIterator.reset();
        }

        long endTime = System.currentTimeMillis();
        log.info("Training completed in " + (endTime - startTime) + " ms");
    }




}
