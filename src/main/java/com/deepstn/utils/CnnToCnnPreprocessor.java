package com.deepstn.utils;


import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

// Custom preprocessor to reshape convolutional to convolutional

public class CnnToCnnPreprocessor implements InputPreProcessor {
    private final int[] newShape;

    public CnnToCnnPreprocessor(int[] newShape) {
        this.newShape = newShape;
    }


    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return output;
    }


    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return input.reshape(newShape);
    }

    public InputPreProcessor clone() {
        try {
            super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return new CnnToCnnPreprocessor(newShape);
    }

    public InputType getOutputType(InputType inputType) {
        return InputType.convolutional(newShape[2], newShape[1], newShape[0]);

    }

    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        return null;
    }


}



