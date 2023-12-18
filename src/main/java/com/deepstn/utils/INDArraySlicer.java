package com.deepstn.utils;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class INDArraySlicer {

    private final INDArray data;

    public INDArraySlicer(INDArray data) {
        this.data = data;
    }

    public INDArray extractSubset(long startSliceIndex, long endSliceIndex) {
        if (endSliceIndex <= startSliceIndex || endSliceIndex > data.size(0)) {
            throw new IllegalArgumentException("Invalid slice indices.");
        }

        // Calculate the size of the subset
        long subsetSize = endSliceIndex - startSliceIndex;
        DataType dataType = data.dataType();


        // Create a new INDArray for the subset with the desired shape
        INDArray subset = Nd4j.create(dataType, subsetSize, data.shape()[1], data.shape()[2], data.shape()[3]);

        // Copy the slices from 'data' to 'subset'
        for (long i = startSliceIndex; i < endSliceIndex; i++) {
            INDArray slice = data.slice(i);
            subset.putSlice((int) (i - startSliceIndex), slice);
        }

        return subset;
    }


}








