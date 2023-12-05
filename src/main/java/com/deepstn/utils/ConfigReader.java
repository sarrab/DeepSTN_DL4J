package com.deepstn.utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

public class ConfigReader {
    private final Properties properties;

    public ConfigReader(String configFilePath) throws IOException {
        properties = new Properties();
        FileInputStream inputStream = new FileInputStream(configFilePath);
        properties.load(inputStream);
    }

    public String getString(String key) {
        return properties.getProperty(key);
    }

    public int getInt(String key) {
        return Integer.parseInt(properties.getProperty(key));
    }

    public double getDouble(String key) {
        return Double.parseDouble(properties.getProperty(key));
    }

    public Properties getProperties() {
        return properties;
    }


}
