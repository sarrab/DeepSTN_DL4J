package com.deepstn.main;

import com.deepstn.dataset.DataManager;
import com.deepstn.utils.ConfigReader;

import java.io.IOException;
import java.util.Properties;


public class MainApp {
    private static ConfigReader configReader;
    private final String datasetPath;

    public MainApp() {
        try {
            configReader = new ConfigReader("src/main/resources/config.properties");
        } catch (IOException e) {
            e.printStackTrace();
        }

        datasetPath = configReader.getString("dataset_path");
    }


    public static void main(String[] args) {
        MainApp mainApp = new MainApp();
        Properties config = MainApp.configReader.getProperties();
        System.out.println(config.getProperty("dataset_path"));
        int lenTest = configReader.getInt("len_test");
        int lenCloseness = configReader.getInt("len_closeness");
        int lenPeriod = configReader.getInt("len_period");
        int lenTrend = configReader.getInt("len_trend");
        int T_closeness = configReader.getInt("T_closeness");
        int T_period = configReader.getInt("T_period");
        int T_trend = configReader.getInt("T_trend");


        DataManager dataManager = new DataManager(mainApp.datasetPath);
        Object[] processedData = dataManager.processData(lenTest, lenCloseness, lenPeriod, lenTrend, T_closeness, T_period, T_trend);


    }
}



