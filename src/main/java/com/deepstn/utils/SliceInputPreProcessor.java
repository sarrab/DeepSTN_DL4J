package com.deepstn.utils;


import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.BaseInputPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SliceInputPreProcessor extends BaseInputPreProcessor {
    private final int startSlice;
    private final int endSlice;

    public SliceInputPreProcessor(int startSlice, int endSlice) {
        this.startSlice = startSlice;
        this.endSlice = endSlice;
    }


    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        // Slicing along the channel dimension (depth)
        return workspaceMgr.dup(ArrayType.ACTIVATIONS, input.get(NDArrayIndex.all(), NDArrayIndex.interval(startSlice, endSlice), NDArrayIndex.all(), NDArrayIndex.all()));
    }

    public INDArray backprop(INDArray indArray, int i, LayerWorkspaceMgr workspaceMgr) {
        // For our slicing operation, the backward pass will just return the input directly.
        return workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, indArray);
    }

    public InputType getOutputType(InputType inputType) {
        if (inputType instanceof InputType.InputTypeConvolutional) {
            InputType.InputTypeConvolutional convInput = (InputType.InputTypeConvolutional) inputType;
            int newDepth = endSlice - startSlice;
            return InputType.convolutional(convInput.getHeight(), convInput.getWidth(), newDepth);
        } else {
            throw new IllegalArgumentException("InputType must be of type InputTypeConvolutional for this preprocessor. Received: " + inputType);
        }

    }


}
