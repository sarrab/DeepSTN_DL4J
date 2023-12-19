package com.deepstn.utils;


import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.BaseInputPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SliceInputPreProcessor extends BaseInputPreProcessor {
    private static final Logger log = LoggerFactory.getLogger(SliceInputPreProcessor.class);
    private final int startSlice;
    private final int endSlice;
    private final int totalChannels;

    public SliceInputPreProcessor(int startSlice, int endSlice, int totalChannels) {
        this.startSlice = startSlice;
        this.endSlice = endSlice;
        this.totalChannels = totalChannels;
    }


    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        // Slicing along the channel dimension
        return workspaceMgr.dup(ArrayType.ACTIVATIONS, input.get(NDArrayIndex.all(), NDArrayIndex.interval(startSlice, endSlice), NDArrayIndex.all(), NDArrayIndex.all()));
    }

    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        long height = output.size(2);
        long width = output.size(3);
        long[] originalShape = new long[]{miniBatchSize, totalChannels, height, width};

        INDArray backpropGradient = Nd4j.zeros(originalShape);

        backpropGradient.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(startSlice, endSlice), NDArrayIndex.all(), NDArrayIndex.all()}, output);

        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, backpropGradient);
    }

    public InputType getOutputType(InputType inputType) {
        if (inputType instanceof InputType.InputTypeConvolutional convInput) {
            int newDepth;
            newDepth = endSlice - startSlice;
            return InputType.convolutional(convInput.getHeight(), convInput.getWidth(), newDepth);
        } else {
            throw new IllegalArgumentException("InputType must be of type InputTypeConvolutional for this preprocessor. Received: " + inputType);
        }

    }


}
