package com.deepstn.main;

import com.deepstn.dataset.DataManager;
import com.deepstn.training.DSTNPlusTrainer;
import com.deepstn.utils.ConfigReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;


public class MainApp {
    private static final Logger log = LoggerFactory.getLogger(MainApp.class);

    private static ConfigReader configReader;
    private static String datasetPath;


    public static void main(String[] args) {

        try {
            configReader = new ConfigReader("src/main/resources/config.properties");
        } catch (IOException e) {
            e.printStackTrace();
        }

        datasetPath = configReader.getString("datasetPath");
        Properties config = configReader.getProperties();
        int lenTest = configReader.getInt("len_test");
        int lenCloseness = configReader.getInt("len_closeness");
        int lenPeriod = configReader.getInt("len_period");
        int lenTrend = configReader.getInt("len_trend");
        int T_closeness = configReader.getInt("T_closeness");
        int T_period = configReader.getInt("T_period");
        int T_trend = configReader.getInt("T_trend");


        DataManager dataManager = new DataManager(datasetPath);
        Object[] processedData = dataManager.processData(lenTest, lenCloseness, lenPeriod, lenTrend, T_closeness, T_period, T_trend);
        DSTNPlusTrainer trainer = new DSTNPlusTrainer(processedData, config);
        trainer.train();


    }
}



