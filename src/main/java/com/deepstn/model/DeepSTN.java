package com.deepstn.model;

import com.deepstn.utils.SliceInputPreProcessor;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DeepSTN {
    private static final Logger log = LoggerFactory.getLogger(DeepSTN.class);

    public String convUnit(ComputationGraphConfiguration.GraphBuilder graphBuilder, String lastLayerName, String where, int kernel_size, int fin, int fout, double drop) {
        graphBuilder
                .addLayer("relu" + where, new ActivationLayer.Builder().activation(Activation.RELU).build(), lastLayerName)
                .addLayer("bn" + where, new BatchNormalization(), "relu" + where)
                .addLayer("dropout" + where, new DropoutLayer(drop), "bn" + where)
                .addLayer("conv2D" + where, new ConvolutionLayer.Builder(kernel_size, kernel_size)
                        .nOut(fout)
                        .nIn(fin)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "dropout" + where);

        return "conv2D" + where;

    }


    public String createResPlusEModel(ComputationGraphConfiguration.GraphBuilder graphBuilder, String lastLayerName, int i, int f, int fPlus, int rate, double drop, int H, int W) {

        String layerName;
        if (rate == 1) {
            layerName = lastLayerName;
        } else {
            graphBuilder.addLayer("pool_resPlus" + i, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                    .stride(rate, rate)
                    .kernelSize(rate, rate)


                    .build(), lastLayerName);
            layerName = "pool_resPlus" + i;
        }
        graphBuilder.addLayer("relu_resPlus" + i, new ActivationLayer.Builder().activation(Activation.RELU).build(), layerName)
                .addLayer("bn_resPlus" + i, new BatchNormalization.Builder().build(), "relu_resPlus" + i);
        graphBuilder.addLayer("conv2d_resPlus" + i, new ConvolutionLayer.Builder()
                .kernelSize((int) (Math.floor(H / (double) (rate))), (int) (Math.floor(W / (double) (rate))))
                .nOut(fPlus * H * W)

                .padding(0, 0)
                .build(), "bn_resPlus" + i);


        ReshapeVertex reshapeVertex = new ReshapeVertex(-1, fPlus, H, W);
        graphBuilder.addVertex("reshape_resPlus" + i, reshapeVertex, "conv2d_resPlus" + i);


        String convUnit = convUnit(graphBuilder, lastLayerName, "convU1_resPlus" + i, 3, f, f - fPlus, drop);

        graphBuilder.addVertex("concat_resPlus" + i, new MergeVertex(), convUnit, ("reshape_resPlus" + i));

        String convUnit2 = convUnit(graphBuilder, "concat_resPlus" + i, "convU2_resPlus" + i, 3, f, f, drop);
        String resPlusOutput = "resPlusOutput" + i;
        graphBuilder.addVertex(resPlusOutput, new ElementWiseVertex(ElementWiseVertex.Op.Add), lastLayerName, convUnit2);
        return resPlusOutput;

    }

    public String timeTransform(ComputationGraphConfiguration.GraphBuilder graphBuilder, String lastLayer, int poiIndex, int T, int T_F) {
        graphBuilder.addLayer("poi_" + poiIndex + "conv2d_TF_1", new ConvolutionLayer.Builder(1, 1)
                        .nOut(T_F)
                        .nIn(T + 7)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), lastLayer)
                .addLayer("poi_" + poiIndex + "relu_TF_1", new ActivationLayer.Builder().activation(Activation.RELU).build(), "poi_" + poiIndex + "conv2d_TF_1")
                .addLayer("poi_" + poiIndex + "conv2d_TF_2", new ConvolutionLayer.Builder(1, 1)
                        .nOut(1)
                        .nIn(T + 7)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "poi_" + poiIndex + "relu_TF_1")
                .addLayer("poi_" + poiIndex + "relu_TF_2", new ActivationLayer.Builder().activation(Activation.RELU).build(), "poi_" + poiIndex + "conv2d_TF_2");
        return "poi_" + poiIndex + "relu_TF_2";


    }

    public String poiTimeUnit(ComputationGraphConfiguration.GraphBuilder graphBuilder, String poiInput, String timeInput, int P_N, int PT_F, int T, int T_F, int H, int W, boolean isPT_F) {
        String[] timeTransformedLayers = new String[P_N];


        for (int i = 0; i < P_N; i++) {
            timeTransformedLayers[i] = timeTransform(graphBuilder, timeInput, i, T, T_F);

        }

        graphBuilder.addVertex("concatenateTime", new MergeVertex(), timeTransformedLayers);

        graphBuilder.addVertex("mult_time_poi", new ElementWiseVertex(ElementWiseVertex.Op.Product), poiInput, "concatenateTime");
        String poiTimeOutput = "mult_time_poi";
        if (isPT_F) {
            graphBuilder.addLayer("PT_moreConv", new ConvolutionLayer.Builder(1, 1)
                    .nOut(PT_F)
                    .convolutionMode(ConvolutionMode.Same)
                    .build(), "mult_time_poi");
            poiTimeOutput = "PT_moreConv";

        }

        return poiTimeOutput;
    }


    public ComputationGraph buildModel(int H, int W, int channel, int c, int p, int t,
                                       int pre_F, int conv_F, int R_N,
                                       int plus, int rate, boolean is_pt, int P_N, int T_F, int PT_F, int T,
                                       double drop, double lr, int kernel_size_early_fusion, boolean isPT_F, long seed) {
        int all_channel = channel * (c + p + t);

        int close_channel = channel * c;
        int period_channel = channel * p;
        int trend_channel = channel * t;

        // Slicing using custom InputPreProcessor
        SliceInputPreProcessor cPreProcessor = new SliceInputPreProcessor(0, close_channel, all_channel);
        SliceInputPreProcessor pPreProcessor = new SliceInputPreProcessor(close_channel, close_channel + period_channel, all_channel);
        SliceInputPreProcessor tPreProcessor = new SliceInputPreProcessor(close_channel + period_channel, close_channel + period_channel + trend_channel, all_channel);


        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr)) // Adam optimizer
                .graphBuilder()
                .addInputs(is_pt ? new String[]{"flow_input", "poi_input", "time_input"} : new String[]{"flow_input"})
                .setInputTypes(is_pt ? new InputType[]{InputType.convolutional(H, W, all_channel)
                        , InputType.convolutional(H, W, P_N), InputType.convolutional(H, W, T + 7)} : new InputType[]{InputType.convolutional(H, W, all_channel)})


                .addLayer("closeConv", new ConvolutionLayer.Builder(1, 1)
                        .nIn(close_channel).nOut(pre_F)
                        .build(), cPreProcessor, "flow_input")
                .addLayer("periodConv", new ConvolutionLayer.Builder(1, 1)
                        .nIn(period_channel).nOut(pre_F)
                        .build(), pPreProcessor, "flow_input")
                .addLayer("trendConv", new ConvolutionLayer.Builder(1, 1)
                        .nIn(trend_channel).nOut(pre_F)
                        .build(), tPreProcessor, "flow_input");

        String lastLayer;

        if (is_pt) {
            String poiTime = poiTimeUnit(graphBuilder, "poi_input", "time_input", P_N, PT_F, T, T_F, H, W, isPT_F);
            graphBuilder.addVertex("concat_c_p_t_tpoi", new MergeVertex(), "closeConv", "periodConv", "trendConv", poiTime);
            lastLayer = convUnit(graphBuilder, "concat_c_p_t_tpoi", "_eFusion_main", kernel_size_early_fusion, isPT_F ? pre_F * 3 + PT_F : pre_F * 3 + P_N, conv_F, drop);

        } else {
            graphBuilder.addVertex("concat_c_p_t", new MergeVertex(), "closeConv", "periodConv", "trendConv");
            lastLayer = convUnit(graphBuilder, "concat_c_p_t", "_eFusion_main", kernel_size_early_fusion, pre_F * 3, conv_F, drop);


        }


        for (int i = 0; i < R_N; i++) {

            lastLayer = createResPlusEModel(graphBuilder, lastLayer, i + 1, conv_F, plus, rate, drop, H, W);

        }


        graphBuilder.addLayer("relu", new ActivationLayer.Builder().activation(Activation.RELU).build(), lastLayer)
                .addLayer("bn", new BatchNormalization(), "relu");
        graphBuilder.addLayer("dropout", new DropoutLayer(drop), "bn");
        graphBuilder.addLayer("conv2d", new ConvolutionLayer.Builder()//
                        .kernelSize(1, 1)
                        .padding(0, 0)
                        .nIn(conv_F)
                        .nOut(channel)//
                        .build(), "dropout")
                .addLayer("tanh", new ActivationLayer.Builder().activation(Activation.TANH).build(), "conv2d")
                .addLayer("lossLayer", new CnnLossLayer.Builder(LossFunctions.LossFunction.MSE).build(), "tanh")
                .setOutputs("lossLayer").build();


        ComputationGraph model = new ComputationGraph(graphBuilder.build());
        model.init();
        log.info(model.summary());

        return model;
    }
}
