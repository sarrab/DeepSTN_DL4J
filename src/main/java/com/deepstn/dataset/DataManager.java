package com.deepstn.dataset;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class DataManager {
    private final INDArray allData;
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
        System.out.println("******************Loading data from" + datasetPath + "**********************");
        return Nd4j.readNpy(datasetPath);

    }

    private INDArray normalizeData() {

        return allData.sub(minVal).div(maxVal - minVal).mul(2).sub(1);
    }

    private INDArray computeTemporalFeatures(int T_period) {
        INDArray time = Nd4j.arange(lenTotal);
        INDArray timeHour = time.fmod(T_period);
        INDArray timeDay = time.div(T_period).fmod(7);


        INDArray matrixHour = Nd4j.zeros(lenTotal, 24, allData.size(2), allData.size(3));
        INDArray matrixDay = Nd4j.zeros(lenTotal, 7, allData.size(2), allData.size(3));

        for (int i = 0; i < lenTotal; i++) {
            int hourIndex = timeHour.getInt(i);
            // set the corresponding slices to 1
            matrixHour.get(NDArrayIndex.point(i), NDArrayIndex.point(hourIndex), NDArrayIndex.all(), NDArrayIndex.all()).assign(1);
            int dayIndex = timeDay.getInt(i);
            // set the corresponding slices to 1
            matrixDay.get(NDArrayIndex.point(i), NDArrayIndex.point(dayIndex), NDArrayIndex.all(), NDArrayIndex.all()).assign(1);
        }
        INDArray matrixT = Nd4j.concat(1, matrixHour, matrixDay);


        return matrixT;
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
            System.out.println("Wrong");
        }
        return numberOfSkipHours;
    }

    private INDArray prepareTimeFeatureData(int timeFeatureLen, int timeFeatureUnit, int numberOfSkipHours) {
        INDArray tFeatureData;
        tFeatureData = allData.get(NDArrayIndex.interval(numberOfSkipHours - timeFeatureUnit, lenTotal - timeFeatureUnit),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        for (int i = 0; i < timeFeatureLen - 1; i++) {
            INDArray slice = allData.get(NDArrayIndex.interval(numberOfSkipHours - timeFeatureUnit * (2 + i),
                            lenTotal - timeFeatureUnit * (2 + i)),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            tFeatureData = Nd4j.concat(1, tFeatureData, slice);
        }
        return tFeatureData;
    }


    public Object[] processData(int lenTest, int lenCloseness, int lenPeriod, int lenTrend, int TCloseness, int TPeriod, int TTrend) {
        INDArray allData = normalizeData();

        int numberOfSkipHours = determineSkipHours(lenCloseness, lenPeriod, lenTrend, TCloseness, TPeriod, TTrend);
        INDArray matrixT = computeTemporalFeatures(TPeriod);

        INDArray xCloseness = null, xPeriod = null, xTrend = null;
        INDArray y = allData.get(NDArrayIndex.interval(numberOfSkipHours, lenTotal),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

        //Preparation of Closeness data
        if (lenCloseness > 0)
            xCloseness = prepareTimeFeatureData(lenCloseness, TCloseness, numberOfSkipHours);

        //Preparation of Period data
        if (lenPeriod > 0)
            xPeriod = prepareTimeFeatureData(lenPeriod, TPeriod, numberOfSkipHours);

        //Preparation of Trend data
        if (lenTrend > 0)
            xTrend = prepareTimeFeatureData(lenTrend, TTrend, numberOfSkipHours);

        // Splitting Data (train/test):

        matrixT = matrixT.get(NDArrayIndex.interval(numberOfSkipHours, matrixT.size(0)), NDArrayIndex.all());

        // Splitting Closeness Data:
        INDArray xClosenessTrain = xCloseness.get(NDArrayIndex.interval(0, xCloseness.size(0) - lenTest),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray xClosenessTest = xCloseness.get(NDArrayIndex.interval(xCloseness.size(0) - lenTest, xCloseness.size(0)),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

        //Splitting Period Data:
        INDArray xPeriodTrain = xPeriod.get(NDArrayIndex.interval(0, xPeriod.size(0) - lenTest),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray xPeriodTest = xPeriod.get(NDArrayIndex.interval(xPeriod.size(0) - lenTest, xPeriod.size(0)),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

        //Splitting Trend Data:
        INDArray xTrendTrain = xTrend.get(NDArrayIndex.interval(0, xTrend.size(0) - lenTest), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray xTrendTest = xTrend.get(NDArrayIndex.interval(xTrend.size(0) - lenTest, xTrend.size(0)),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());


        // Splitting y Data:

        INDArray yTrain = y.get(NDArrayIndex.interval(0, y.size(0) - lenTest), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray yTest = y.get(NDArrayIndex.interval(y.size(0) - lenTest, y.size(0)), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());

        // Splitting time Data:
        INDArray timeTrain = matrixT.get(NDArrayIndex.interval(0, matrixT.size(0) - lenTest), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray timeTest = matrixT.get(NDArrayIndex.interval(matrixT.size(0) - lenTest, matrixT.size(0)), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray xTrain = Nd4j.concat(1, xClosenessTrain, xPeriodTrain, xTrendTrain);
        INDArray xTest = Nd4j.concat(1, xClosenessTest, xPeriodTest, xTrendTest);

        int lenTrain = (int) xClosenessTrain.size(0);
        lenTest = (int) xClosenessTest.size(0);

        System.out.println("lenTrain: " + lenTrain);
        System.out.println("lenTest: " + lenTest);


        return new Object[]{xTrain, timeTrain, yTrain, xTest, timeTest, yTest, maxVal - minVal};
    }
}




