package com.deepstn.model;

import com.deepstn.utils.SliceInputPreProcessor;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.util.IdentityLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class DeepSTN {


    public String convUnit(ComputationGraphConfiguration.GraphBuilder graphBuilder, String lastLayerName, int kernel_size, int fin, int fout, double drop, int h, int w, long seed) {
        graphBuilder.setInputTypes(InputType.convolutional(h, w, fin))
                .addLayer("relu", new ActivationLayer.Builder().activation(Activation.RELU).build(), lastLayerName)
                .addLayer("bn", new BatchNormalization(), "relu")
                .addLayer("dropout", new DropoutLayer(drop), "bn")
                .addLayer("conv2D_" + kernel_size, new ConvolutionLayer.Builder(kernel_size, kernel_size)
                        .nOut(fout)
                        .padding(1, 1)
                        .build(), "dropout")
                .build();


        return "convUnit_ksize_" + kernel_size;

    }


    public String createResPlusEModel(ComputationGraphConfiguration.GraphBuilder graphBuilder, String lastLayerName, int F, int Fplus, int rate, double drop, int H, int W, long seed) {
        String layerName;
        if (rate == 1) {
            layerName = lastLayerName;
        } else {
            graphBuilder.addLayer("pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                    .stride(rate, rate)
                    .kernelSize(rate, rate)
                    .build(), lastLayerName);
            layerName = "pool";
        }
        graphBuilder.addLayer("relu", new ActivationLayer.Builder().activation(Activation.RELU).build(), layerName)
                .addLayer("bn", new BatchNormalization(), "relu")
                .addLayer("conv2d", new ConvolutionLayer.Builder()
                        .kernelSize((int) (Math.floor(H / (double) (rate))), (int) (Math.floor(H / (double) (rate))))
                        .nOut(Fplus * H * W)
                        .padding(0, 0)  // "valid" padding means no padding in Keras/TensorFlow, we can omit this in DL4J
                        .build());
        String convUnit = convUnit(graphBuilder, "conv2d", 3, F, F - Fplus, drop, H, W, seed);
        graphBuilder.addVertex("reshape", new ReshapeVertex(Fplus, H, W), "conv2d");
        graphBuilder.addVertex("concat", new MergeVertex(), convUnit, "reshape");
        String convUnit2 = convUnit(graphBuilder, "concat", 3, F, F, drop, H, W, seed) + 2;
        String resPlusOutput = "resPlusOutput";
        graphBuilder.addVertex(resPlusOutput, new ElementWiseVertex(ElementWiseVertex.Op.Add), lastLayerName, convUnit2)
                .setInputTypes(InputType.convolutional(H, W, F))
                .build();
        return resPlusOutput;

    }


    public ComputationGraph buildModel(int H, int W, int channel, int c, int p, int t,
                                       int pre_F, int conv_F, int R_N, boolean is_plus, boolean is_plus_efficient,
                                       int plus, int rate, boolean is_pt, int P_N, int T_F, int PT_F, int T,
                                       double drop, boolean is_summary, double lr, int kernel_size_early_fusion, boolean isPT_F, long seed) {
        int all_channel = channel * (c + p + t);

        int close_channel = channel * c;
        int period_channel = channel * p;
        int trend_channel = channel * t;
        // Slicing using custom InputPreProcessor
        SliceInputPreProcessor cPreProcessor = new SliceInputPreProcessor(0, close_channel);
        SliceInputPreProcessor pPreProcessor = new SliceInputPreProcessor(close_channel, close_channel + period_channel);
        SliceInputPreProcessor tPreProcessor = new SliceInputPreProcessor(close_channel + period_channel, close_channel + period_channel + trend_channel);


        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr)) // Adam optimizer
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(H, W, channel))
//                .addVertex("closeInput", new ReshapeVertex(new int[]{H,W,close_channel}), "input")
//                .addVertex("periodInput", new ReshapeVertex(new int[]{H, W, period_channel}), "input")
//                .addVertex("trendInput", new ReshapeVertex(new int[]{H, W, trend_channel}), "input")
                .addLayer("closeConv", new ConvolutionLayer.Builder(1, 1)
                        .nIn(close_channel).nOut(conv_F)
                        .build(), cPreProcessor, "input")
                .addLayer("periodConv", new ConvolutionLayer.Builder(1, 1)
                        .nIn(period_channel).nOut(conv_F)
                        .build(), pPreProcessor, "input")
                .addLayer("trendConv", new ConvolutionLayer.Builder(1, 1)
                        .nIn(trend_channel).nOut(conv_F)
                        .build(), tPreProcessor, "input")
                .addVertex("concat_c_p_t", new MergeVertex(), "closeConv", "periodConv", "trendConv");
        String early_fusion = convUnit(graphBuilder, "concat_c_p_t", kernel_size_early_fusion, pre_F * 3, conv_F, drop, H, W, seed);
        graphBuilder.addLayer("earlyFusion", new IdentityLayer(), early_fusion)
                .build();
        String lastLayer = "earlyFusion";
        if (is_plus) {
            if (is_plus_efficient) {
                //TODO;
            } else {
                for (int i = 0; i < R_N; i++) {
                    createResPlusEModel(graphBuilder, lastLayer, conv_F, plus, rate, drop, H, W, seed);
                    lastLayer = "resPlusE" + (i + 1);

                }
            }
        } else {
            //TODO;
        }

        graphBuilder.addLayer("relu", new ActivationLayer.Builder().activation(Activation.RELU).build(), lastLayer)
                .addLayer("bn", new BatchNormalization(), "relu")
                .addLayer("dropout", new DropoutLayer(drop), "bn")
                .addLayer("conv2d", new ConvolutionLayer.Builder()
                        .kernelSize(1, 1)
                        .padding(0, 0)  // "valid" padding means no padding in Keras/TensorFlow
                        .nOut(channel)
                        .build(), "dropout")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.TANH)
                        .nOut(channel)
                        .build(), "conv2d")
                .setOutputs("output")
                .setInputTypes(InputType.convolutional(H, W, channel))
                .build();

        ComputationGraph model = new ComputationGraph(graphBuilder.build());
        model.init();
        if (is_summary) {
            System.out.println(model.summary());
        }
        return model;
    }
}
