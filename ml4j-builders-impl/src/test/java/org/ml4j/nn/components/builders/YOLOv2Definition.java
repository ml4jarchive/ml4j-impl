package org.ml4j.nn.components.builders;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons3D;

public class YOLOv2Definition implements Component3Dto3DGraphDefinition {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	public static final ActivationFunctionType LEAKY_RELU_ACTIVATION_FUNCTION_TYPE 
		= ActivationFunctionType.getBaseType(ActivationFunctionBaseType.LEAKYRELU);
	
	private YOLOv2WeightsLoader weightsLoader;
	
	public YOLOv2Definition(YOLOv2WeightsLoader weightsLoader) {
		this.weightsLoader = weightsLoader;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(608, 608, 3, false);
	}
	
	@Override
	public Neurons3D getOutputNeurons() {
		return new Neurons3D(19, 19, 425, false);
	}

	@Override
	public <T extends NeuralComponent> InitialComponents3DGraphBuilder<T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start, NeuralComponentFactory<T> neuralComponentFactory) {
		
		NeuralComponentType spaceToDepthComponentType
			= NeuralComponentType.createSubType(NeuralComponentBaseType.AXONS, "SPACE_TO_DEPTH");
		
		// input_1
		return start
				// conv2d_1
				.withConvolutionalAxons("conv2d_1")
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_1_kernel0", 3, 3, 3, 32))
				.withFilterSize(3, 3)
				.withFilterCount(32)
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(608, 608, 32, false))
				// batch_normalization_1
				.withBatchNormAxons("batch_normalization_1")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_1_moving_mean0", 32))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_1_moving_variance0", 32))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_1_gamma0", 32))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_1_beta0", 32))
				.withConnectionToNeurons(new Neurons3D(608, 608, 32, false))
				.withActivationFunction("leaky_re_lu_1", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties()
						.withAlpha(0.1f))
				// max_pooling2d_1
				.withMaxPoolingAxons("max_pooling2d_1")
				.withFilterSize(2, 2)
				.withStride(2, 2)
				.withConnectionToNeurons(new Neurons3D(304, 304, 32, false))
				// conv2d_2
				.withConvolutionalAxons("conv2d_2")
				.withFilterSize(3, 3)
				.withFilterCount(64)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_2_kernel0", 3, 3, 32, 64))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(304, 304, 64, false))
				// batch_normalization_2
				.withBatchNormAxons("batch_normalization_2")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_2_moving_mean0", 64))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_2_moving_variance0", 64))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_2_gamma0", 64))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_2_beta0", 64))
				.withConnectionToNeurons(new Neurons3D(304, 304, 64, false))
				.withActivationFunction("leaky_re_lu_2", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, 
						new ActivationFunctionProperties().withAlpha(0.1f))
				// max_pooling2d_2
				.withMaxPoolingAxons("max_pooling2d_2")
				.withFilterSize(2, 2)
				.withStride(2, 2)
				.withConnectionToNeurons(new Neurons3D(152, 152, 64, false))
				// conv2d_3
				.withConvolutionalAxons("conv2d_3")
				.withFilterSize(3, 3)
				.withFilterCount(128)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_3_kernel0", 3, 3, 64, 128))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				// batch_normalization_3
				.withBatchNormAxons("batch_normalization_3")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_3_moving_mean0", 128))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_3_moving_variance0", 128))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_3_gamma0", 128))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_3_beta0", 128))
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				.withActivationFunction("leaky_relu_3", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, 
						new ActivationFunctionProperties().withAlpha(0.1f))
				// conv2d_4
				.withConvolutionalAxons("conv2d_4")
				.withFilterSize(1, 1)
				.withFilterCount(128)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_4_kernel0", 1, 1, 128, 64))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(152, 152, 64, false))
				// batch_normalization_4
				.withBatchNormAxons("batch_normalization_4")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_4_moving_mean0", 64))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_4_moving_variance0", 64))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_4_gamma0", 64))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_4_beta0", 64))
				.withConnectionToNeurons(new Neurons3D(152, 152, 64, false))
				.withActivationFunction("leaky_re_lu_4", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				// conv2d_5
				.withConvolutionalAxons("conv2d_5")
				.withFilterSize(3, 3)
				.withFilterCount(128)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_5_kernel0", 3, 3, 64, 128))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				// batch_normalization_5
				.withBatchNormAxons("batch_normalization_5")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_5_moving_mean0", 128))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_5_moving_variance0", 128))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_5_gamma0", 128))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_5_beta0", 128))
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				.withActivationFunction("leaky_re_lu_5", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				.withMaxPoolingAxons("max_pooling2d_3")
				.withFilterSize(2, 2)
				.withStride(2, 2)
				.withConnectionToNeurons(new Neurons3D(76, 76, 128, false))
				// conv2d_6
				.withConvolutionalAxons("conv2d_6")
				.withFilterSize(3, 3)
				.withFilterCount(256)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_6_kernel0", 3, 3, 128, 256))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				// batch_normalization_6
				.withBatchNormAxons("batch_normalization_6")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_6_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_6_moving_variance0", 256))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_6_gamma0", 256))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_6_beta0", 256))
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				.withActivationFunction("leaky_re_lu_6", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				// conv2d_7
				.withConvolutionalAxons("conv2d_7")
				.withFilterSize(1, 1)
				.withFilterCount(128)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_7_kernel0", 1, 1, 256, 128))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(76, 76, 128, false))
				// batch_normalization_7
				.withBatchNormAxons("batch_normalization_7")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_7_moving_mean0", 128))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_7_moving_variance0", 128))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_7_gamma0", 128))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_7_beta0", 128))
				.withConnectionToNeurons(new Neurons3D(76, 76, 128, false))
				.withActivationFunction("leaky_re_lu_7", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				// conv2d_8
				.withConvolutionalAxons("conv2d_8")
				.withFilterSize(3, 3)
				.withFilterCount(256)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_8_kernel0", 3, 3, 128, 256))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				// batch_normalization_8
				.withBatchNormAxons("batch_normalization_8")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_8_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_8_moving_variance0", 256))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_8_gamma0", 256))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_8_beta0", 256))
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				.withActivationFunction("leaky_re_lu_8", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				.withMaxPoolingAxons("max_pooling2d_4")
				.withFilterSize(2, 2)
				.withStride(2, 2)
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				// conv2d_9
				.withConvolutionalAxons("conv2d_9")
				.withFilterSize(3, 3)
				.withFilterCount(512)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_9_kernel0", 3, 3, 256, 512))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				// batch_normalization_9
				.withBatchNormAxons("batch_normalization_9")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_9_moving_mean0", 512))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_9_moving_variance0", 512))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_9_gamma0", 512))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_9_beta0", 512))
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withActivationFunction("leaky_re_lu_9", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				// conv2d_10
				.withConvolutionalAxons("conv2d_10")
				.withFilterSize(1, 1)
				.withFilterCount(256)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_10_kernel0", 1, 1, 512, 256))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				// batch_normalization_10
				.withBatchNormAxons("batch_normalization_10")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_10_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_10_moving_variance0", 256))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_10_gamma0", 256))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_10_beta0", 256))
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				.withActivationFunction("leaky_re_lu_10", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				// conv2d_11
				.withConvolutionalAxons("conv2d_11")
				.withFilterSize(3, 3)
				.withFilterCount(512)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_11_kernel0", 3, 3, 256, 512))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				// batch_normalization_11
				.withBatchNormAxons("batch_normalization_11")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_11_moving_mean0", 512))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_11_moving_variance0", 512))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_11_gamma0", 512))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_11_beta0", 512))
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withActivationFunction("leaky_re_lu_11", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				// conv2d_12
				.withConvolutionalAxons("conv2d_12")
				.withFilterSize(1, 1)
				.withFilterCount(256)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_12_kernel0", 1, 1, 512, 256))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				// batch_normalization_12
				.withBatchNormAxons("batch_normalization_12")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_12_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_12_moving_variance0", 256))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_12_gamma0", 256))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_12_beta0", 256))
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				.withActivationFunction("leaky_re_lu_12", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				// conv2d_13
				.withConvolutionalAxons("conv2d_13")
				.withFilterSize(3, 3)
				.withFilterCount(512)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_13_kernel0", 3, 3, 256, 512))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withBatchNormAxons("batch_normalization_13")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_13_moving_mean0", 512))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_13_moving_variance0", 512))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_13_gamma0", 512))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_13_beta0", 512))
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withActivationFunction("leaky_re_lu_13", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				.withParallelPaths()
				.withPath()
					// conv2d_21
					.withConvolutionalAxons("conv2d_21")
					.withFilterSize(1, 1)
					.withFilterCount(64)
					.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_21_kernel0", 1, 1, 512, 64))
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(38, 38, 64, false))
					// batch_normalization_21
					.withBatchNormAxons("batch_normalization_21")
					.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_21_moving_mean0", 64))
					.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_21_moving_variance0", 64))
					.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_21_gamma0", 64))
					.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_21_beta0", 64))
					.withConnectionToNeurons(new Neurons3D(38, 38, 64, false))
					// leaky_re_lu_21
					.withActivationFunction("leaky_re_lu_21", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
					// space_to_depth_x2
					.with3DComponent(neuralComponentFactory.createComponent("space_to_depth_x2", new Neurons3D(38, 38, 64, false), 
							new Neurons3D(19, 19, 256, false), spaceToDepthComponentType), new Neurons3D(19, 19, 256, false))
					.endPath()
				.withPath()
					.withMaxPoolingAxons("max_pooling2d_5")
					.withFilterSize(2, 2)
					.withStride(2, 2)
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					// conv2d_14
					.withConvolutionalAxons("conv2d_14")
					.withFilterSize(3, 3)
					.withFilterCount(1024)
					.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_14_kernel0", 3, 3, 512, 1024))
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_14
					.withBatchNormAxons("batch_normalization_14")
					.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_14_moving_mean0", 1024))
					.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_14_moving_variance0", 1024))
					.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_14_gamma0", 1024))
					.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_14_beta0", 1024))
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					.withActivationFunction("leaky_re_lu_14", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
					// conv2d_15
					.withConvolutionalAxons("conv2d_15")
					.withFilterSize(1, 1)
					.withFilterCount(512)
					.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_15_kernel0", 1, 1, 1024, 512))
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					// batch_normalization_15
					.withBatchNormAxons("batch_normalization_15")
					.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_15_moving_mean0", 512))
					.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_15_moving_variance0", 512))
					.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_15_gamma0", 512))
					.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_15_beta0", 512))
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					.withActivationFunction("leaky_re_lu_15", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
					// conv2d_16
					.withConvolutionalAxons("conv2d_16")
					.withFilterSize(3, 3)
					.withFilterCount(1024)
					.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_16_kernel0", 3, 3, 512, 1024))
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_16
					.withBatchNormAxons("batch_normalization_16")
					.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_16_moving_mean0", 1024))
					.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_16_moving_variance0", 1024))
					.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_16_gamma0", 1024))
					.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_16_beta0", 1024))
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					.withActivationFunction("leaky_re_lu_16", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
					// conv2d_17
					.withConvolutionalAxons("conv2d_17")
					.withFilterSize(1, 1)
					.withFilterCount(512)
					.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_17_kernel0", 1, 1, 1024, 512))
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					// batch_normalization_17
					.withBatchNormAxons("batch_normalization_17")
					.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_17_moving_mean0", 512))
					.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_17_moving_variance0", 512))
					.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_17_gamma0", 512))
					.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_17_beta0", 512))
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					.withActivationFunction("leaky_re_lu_17", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
					// conv2d_18
					.withConvolutionalAxons("conv2d_18") 
					.withFilterSize(3, 3)
					.withFilterCount(1024)
					.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_18_kernel0", 3, 3, 512, 1024))
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_18
					.withBatchNormAxons("batch_normalization_18")
					.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_18_moving_mean0", 1024))
					.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_18_moving_variance0", 1024))
					.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_18_gamma0", 1024))
					.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_18_beta0", 1024))
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					.withActivationFunction("leaky_re_lu_18", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
					// conv2d_19
					.withConvolutionalAxons("conv2d_19") 
					.withFilterSize(3, 3)
					.withFilterCount(1024)
					.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_19_kernel0", 3, 3, 1024, 1024))
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_19
					.withBatchNormAxons("batch_normalization_19")
					.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_19_moving_mean0", 1024))
					.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_19_moving_variance0", 1024))
					.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_19_gamma0", 1024))
					.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_19_beta0", 1024))
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// leaky_re_lu_19
					.withActivationFunction("leaky_re_lu_19", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
					// conv2d_20
					.withConvolutionalAxons("conv2d_20") 
					.withFilterSize(3, 3)
					.withFilterCount(1024)
					.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_20_kernel0", 3, 3, 1024, 1024))
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_20
					.withBatchNormAxons("batch_normalization_20")
					.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_20_moving_mean0", 1024))
					.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_20_moving_variance0", 1024))
					.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_20_gamma0", 1024))
					.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_20_beta0", 1024))
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// leaky_re_lu_20
					.withActivationFunction("leaky_re_lu_20", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				.endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT)
				// conv2d_22
				.withConvolutionalAxons("conv2d_22")
				.withFilterSize(3, 3)
				.withFilterCount(1024)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_22_kernel0", 3, 3, 1280, 1024))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
				// batch_normalization_22
				.withBatchNormAxons("batch_normalization_22")
				.withMean(weightsLoader.getBatchNormLayerMovingMean("batch_normalization_22_moving_mean0", 1024))
				.withVariance(weightsLoader.getBatchNormLayerMovingVariance("batch_normalization_22_moving_variance0", 1024))
				.withGamma(weightsLoader.getBatchNormLayerWeights("batch_normalization_22_gamma0", 1024))
				.withBeta(weightsLoader.getBatchNormLayerBias("batch_normalization_22_beta0", 1024))
				.withConnectionToNeurons(new Neurons3D(19, 19, 1024, true))
				// leaky_re_lu_20
				.withActivationFunction("leaky_re_lu_22", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties().withAlpha(0.1f))
				// conv2d_23
				.withConvolutionalAxons("conv2d_23") 
				.withFilterSize(1, 1)
				.withFilterCount(425)
				.withStride(1, 1)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_23_kernel0", 1, 1, 1024, 425))
				.withBiasUnit()
				.withBiases(weightsLoader.getConvolutionalLayerBiases("conv2d_23_bias0", 425))
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(19, 19, 425, false));
	}
	
	@Override
	public String getName() {
		return "yolo_v2_graph";
	}

}
