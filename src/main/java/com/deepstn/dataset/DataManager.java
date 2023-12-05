package com.deepstn.dataset;


import com.deepstn.utils.INDArraySlicer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;


public class DataManager {

    private static final Logger log = LoggerFactory.getLogger(DataManager.class);
    private INDArray allData;
    private final double maxVal;
    private final double minVal;
    private final long lenTotal;
    private final long features;
    private final long mapHeight;
    private final long mapWidth;
    private INDArray poi;


    public DataManager(String datasetPath) {
        this.allData = loadData(datasetPath);

        this.lenTotal = allData.size(0);
        this.features = allData.size(1);
        this.mapHeight = allData.size(2);
        this.mapWidth = allData.size(3);
        this.maxVal = allData.maxNumber().doubleValue();
        this.minVal = allData.minNumber().doubleValue();

    }

    public static INDArray loadData(String datasetPath) {
        log.info("****************** Loading data from " + datasetPath + "**********************");
        return Nd4j.readNpy(datasetPath);

    }

    private INDArray normalizeData() {

        log.info("****************** Normalizing data **********************");
        log.info("maxVal " + maxVal);
        log.info("minVal " + minVal);
        return allData.mul(2.0).sub((maxVal + minVal)).div((maxVal - minVal));

    }

    private INDArray computeTemporalFeatures(int T_period) {
        INDArray time = Nd4j.arange(lenTotal);
        INDArray timeHour = time.fmod(T_period);
        INDArray timeDay = time.div(T_period).castTo(DataType.INT64).fmod(7);

        INDArray matrixHour = Nd4j.zeros(lenTotal, 24, mapHeight, mapWidth);
        INDArray matrixDay = Nd4j.zeros(lenTotal, 7, allData.size(2), allData.size(3));

        for (int i = 0; i < lenTotal; i++) {

            int hourIndex = timeHour.getInt(i);


            // set the corresponding slices to 1
            matrixHour.get(NDArrayIndex.point(i), NDArrayIndex.point(hourIndex), NDArrayIndex.all(), NDArrayIndex.all()).assign(1);
            int dayIndex = timeDay.getInt(i);

            // set the corresponding slices to 1
            matrixDay.get(NDArrayIndex.point(i), NDArrayIndex.point(dayIndex), NDArrayIndex.all(), NDArrayIndex.all()).assign(1);
        }
        return Nd4j.concat(1, matrixHour, matrixDay);

    }

    public int determineSkipHours(int lenCloseness, int lenPeriod, int lenTrend, int T_closeness, int T_period, int T_trend) {
        int numberOfSkipHours = 0;
        if (lenTrend > 0) {
            numberOfSkipHours = T_trend * lenTrend;
        } else if (lenPeriod > 0) {
            numberOfSkipHours = T_period * lenPeriod;
        } else if (lenCloseness > 0) {
            numberOfSkipHours = T_closeness * lenCloseness;
        } else {
            log.info("Invalid number of Trend length, Period length or Closeness length");
        }

        log.info("number_of_skip_hours:" + numberOfSkipHours);
        return numberOfSkipHours;
    }

    private INDArray prepareTimeFeatureData(int timeFeatureLen, int timeFeatureUnit, int numberOfSkipHours) {
        INDArray tFeatureData;
        tFeatureData = allData.get(NDArrayIndex.interval(numberOfSkipHours - timeFeatureUnit, lenTotal - timeFeatureUnit),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        for (int i = 0; i < timeFeatureLen - 1; i++) {
            INDArray slice = allData.get(NDArrayIndex.interval((numberOfSkipHours - timeFeatureUnit * (2 + i)),
                            (lenTotal - timeFeatureUnit * (2 + i))),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            tFeatureData = Nd4j.concat(1, tFeatureData, slice);
        }
        return tFeatureData;
    }

    public Object[] processData(int lenTest, int lenCloseness, int lenPeriod, int lenTrend, int TCloseness, int TPeriod, int TTrend) {

        allData = normalizeData();


        int numberOfSkipHours = determineSkipHours(lenCloseness, lenPeriod, lenTrend, TCloseness, TPeriod, TTrend);


        INDArraySlicer slicer = new INDArraySlicer(allData);
        INDArray y = slicer.extractSubset(numberOfSkipHours, lenTotal);


        INDArray matrixT = computeTemporalFeatures(TPeriod);
        matrixT = matrixT.get(NDArrayIndex.interval(numberOfSkipHours, matrixT.size(0)), NDArrayIndex.all());


        //Preparation of Closeness data
        INDArray xCloseness = null, xPeriod = null, xTrend = null;
        if (lenCloseness > 0)
            xCloseness = prepareTimeFeatureData(lenCloseness, TCloseness, numberOfSkipHours);

        //Preparation of Period data
        if (lenPeriod > 0)
            xPeriod = prepareTimeFeatureData(lenPeriod, TPeriod, numberOfSkipHours);

        //Preparation of Trend data
        if (lenTrend > 0)
            xTrend = prepareTimeFeatureData(lenTrend, TTrend, numberOfSkipHours);


        // Splitting Data (train/test):

        // Splitting Closeness Data:
        INDArray xClosenessTrain = Objects.requireNonNull(xCloseness).get(NDArrayIndex.interval(0, xCloseness.size(0) - lenTest),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray xClosenessTest = Objects.requireNonNull(xCloseness).get(NDArrayIndex.interval(xCloseness.size(0) - lenTest, xCloseness.size(0)),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

        //Splitting Period Data:
        INDArray xPeriodTrain = Objects.requireNonNull(xPeriod).get(NDArrayIndex.interval(0, xPeriod.size(0) - lenTest),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray xPeriodTest = Objects.requireNonNull(xPeriod).get(NDArrayIndex.interval(xPeriod.size(0) - lenTest, xPeriod.size(0)),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

        //Splitting Trend Data:
        INDArray xTrendTrain = Objects.requireNonNull(xTrend).get(NDArrayIndex.interval(0, xTrend.size(0) - lenTest), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray xTrendTest = Objects.requireNonNull(xTrend).get(NDArrayIndex.interval(xTrend.size(0) - lenTest, xTrend.size(0)),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());


        // Splitting y Data:
        long size_yTrain = (y.size(0) - lenTest);


        slicer = new INDArraySlicer(y);
        INDArray yTrain = slicer.extractSubset(0, size_yTrain);
        INDArray yTest = slicer.extractSubset(size_yTrain, y.size(0));


        // Splitting time Data:
        INDArray timeTrain = matrixT.get(NDArrayIndex.interval(0, matrixT.size(0) - lenTest), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray timeTest = matrixT.get(NDArrayIndex.interval(matrixT.size(0) - lenTest, matrixT.size(0)), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());


        INDArray xTrain = Nd4j.concat(1, xClosenessTrain, xPeriodTrain, xTrendTrain);
        INDArray xTest = Nd4j.concat(1, xClosenessTest, xPeriodTest, xTrendTest);



        return new Object[]{xTrain, timeTrain, yTrain, xTest, timeTest, yTest, maxVal - minVal};
    }
}






