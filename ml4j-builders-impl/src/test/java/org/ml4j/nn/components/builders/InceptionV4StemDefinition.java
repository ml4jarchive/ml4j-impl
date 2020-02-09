/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.components.builders;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * @author Michael Lavelle
 */
public class InceptionV4StemDefinition implements Component3Dto3DGraphDefinition {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private InceptionV4WeightsLoader weightsLoader;
	private boolean withFreezeOut;
	private float regularisationLambda;
	private float batchNormRegularisationLambda;

	public InceptionV4StemDefinition(InceptionV4WeightsLoader weightsLoader) {
		this.weightsLoader = weightsLoader;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(299, 299, 3, false);
	}
	
	@Override
	public Neurons3D getOutputNeurons() {
		return new Neurons3D(35, 35, 384, false);
	}

	public <T extends NeuralComponent> InitialComponents3DGraphBuilder<T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start, NeuralComponentFactory<T> neuralComponentFactory) {
		return start
				.withConvolutionalAxons("conv2d_1")
					.withConnectionWeights(
							weightsLoader.getConvolutionalLayerWeights("conv2d_1_kernel0", 3, 3, 3, 32))
					.withStride(2, 2).withFilterSize(3, 3).withFilterCount(32).withValidPadding()
					.withAxonsContextConfigurer(
							c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(149, 149, 32, false))
				.withBatchNormAxons("batch_normalization_1")
					.withBiasUnit()
					.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_1_beta0", 32))
					.withMean(weightsLoader.getBatchNormLayerMean(
							"batch_normalization_1_moving_mean0", 32))
					.withVariance(weightsLoader.getBatchNormLayerVariance(
							"batch_normalization_1_moving_variance0", 32))
					.withAxonsContextConfigurer( c -> c.withFreezeOut(withFreezeOut))
					// c.withRegularisationLambda(batchNormRegularisationLambda))
				.withConnectionToNeurons(new Neurons3D(149, 149, 32, false))
				.withActivationFunction("relu_1", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.withConvolutionalAxons("conv2d_2")
					.withConnectionWeights(
							weightsLoader.getConvolutionalLayerWeights("conv2d_2_kernel0", 3, 3, 32, 32))
					.withFilterSize(3, 3).withValidPadding()
					.withAxonsContextConfigurer( c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(147, 147, 32, false))
				.withBatchNormAxons("batch_normalization_2")
					.withBiasUnit()
					.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_2_beta0", 32))
					.withMean(weightsLoader.getBatchNormLayerMean(
							"batch_normalization_2_moving_mean0", 32))
					.withVariance(weightsLoader.getBatchNormLayerVariance(
							"batch_normalization_2_moving_variance0", 32))
					.withAxonsContextConfigurer( c -> c.withFreezeOut(withFreezeOut))
					// .withAxonsContextConfigurer( c ->
					// c.withRegularisationLambda(batchNormRegularisationLambda))
				.withConnectionToNeurons(new Neurons3D(147, 147, 32, false))
					.withActivationFunction("relu_2",ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
					.withConvolutionalAxons("conv2d_3")
						.withConnectionWeights(
								weightsLoader.getConvolutionalLayerWeights("conv2d_3_kernel0", 3, 3, 32, 64))
						.withFilterSize(3, 3).withFilterCount(64).withSamePadding()
						.withAxonsContextConfigurer(
								c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(147, 147, 64, false))
				.withBatchNormAxons("batch_normalization_3")
				.withBiasUnit()
					.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_3_beta0", 64))
					.withMean(weightsLoader.getBatchNormLayerMean(
							"batch_normalization_3_moving_mean0", 64))
					.withVariance(weightsLoader.getBatchNormLayerVariance(
							"batch_normalization_3_moving_variance0", 64))
					.withAxonsContextConfigurer(
							c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(147, 147, 64, false))
				.withActivationFunction("relu_3", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.withParallelPaths()
					.withPath()
						.withMaxPoolingAxons("max_pooling_1")
							.withStride(2, 2)
							.withFilterSize(3, 3)
							.withValidPadding()
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.endPath()
					.withPath()
						.withConvolutionalAxons("conv2d_4")
							.withConnectionWeights(
									weightsLoader.getConvolutionalLayerWeights("conv2d_4_kernel0", 3, 3, 64, 96))
							.withStride(2, 2).withFilterSize(3, 3).withFilterCount(96).withValidPadding()
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 96, false))
						.withBatchNormAxons("batch_normalization_4")
							.withBiasUnit()
							.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_4_beta0", 96))
							.withMean(weightsLoader.getBatchNormLayerMean(
									"batch_normalization_4_moving_mean0", 96))
							.withVariance(weightsLoader.getBatchNormLayerVariance(
									"batch_normalization_4_moving_variance0", 96))
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
							.withConnectionToNeurons(new Neurons3D(73, 73, 96, false))
							.withActivationFunction("relu_4", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
						.endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT)
				.withParallelPaths()
					.withPath()
						.withConvolutionalAxons("conv2d_5")
							.withConnectionWeights(
									weightsLoader.getConvolutionalLayerWeights("conv2d_5_kernel0", 1, 1, 160, 64))
							.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
							.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withBatchNormAxons("batch_normalization_5").withBiasUnit()
							.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_5_beta0", 64))
							.withMean(weightsLoader.getBatchNormLayerMean(
									"batch_normalization_5_moving_mean0", 64))
							.withVariance(weightsLoader.getBatchNormLayerVariance(
									"batch_normalization_5_moving_variance0", 64))
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withActivationFunction("relu_5", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
						.withConvolutionalAxons("conv2d_6")
							.withConnectionWeights(
									weightsLoader.getConvolutionalLayerWeights("conv2d_6_kernel0", 3, 3, 64, 96))
							.withFilterSize(3, 3).withFilterCount(96).withValidPadding()
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(71, 71, 96, false))
						.withBatchNormAxons("batch_normalization_6")
							.withBiasUnit()
							.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_6_beta0",96))
							.withMean(weightsLoader.getBatchNormLayerMean(
									"batch_normalization_6_moving_mean0", 96))
							.withVariance(weightsLoader.getBatchNormLayerVariance(
									"batch_normalization_6_moving_variance0",96))
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(71, 71, 96, false))
						.withActivationFunction("relu_6",ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
					.endPath()
					.withPath()
						.withConvolutionalAxons("conv2d_7")
							.withConnectionWeights(
									weightsLoader.getConvolutionalLayerWeights("conv2d_7_kernel0", 1, 1, 160, 64))
							.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withBatchNormAxons("batch_normalization_7")
							.withBiasUnit()
							.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_7_beta0", 64))
							.withMean(weightsLoader.getBatchNormLayerMean(
									"batch_normalization_7_moving_mean0", 64))
							.withVariance(weightsLoader.getBatchNormLayerVariance(
									"batch_normalization_7_moving_variance0", 64))
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withActivationFunction("relu_7", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
						.withConvolutionalAxons("conv2d_8")
							.withConnectionWeights(
									weightsLoader.getConvolutionalLayerWeights("conv2d_8_kernel0", 7, 1, 64, 64))
							.withFilterSize(7, 1).withFilterCount(64).withSamePadding()
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
							.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withBatchNormAxons("batch_normalization_8")
							.withBiasUnit()
							.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_8_beta0", 64))
							.withMean(weightsLoader.getBatchNormLayerMean(
									"batch_normalization_8_moving_mean0", 64))
							.withVariance(weightsLoader.getBatchNormLayerVariance(
									"batch_normalization_8_moving_variance0", 64))
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withActivationFunction("relu_8", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
						.withConvolutionalAxons("conv2d_9")
							.withConnectionWeights(
									weightsLoader.getConvolutionalLayerWeights("conv2d_9_kernel0", 1, 7, 64, 64))
							.withFilterSize(1, 7).withFilterCount(64).withSamePadding()
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withBatchNormAxons("batch_normalization_9")
						.withBiasUnit()
							.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_9_beta0", 64))
							.withMean(weightsLoader.getBatchNormLayerMean(
									"batch_normalization_9_moving_mean0", 64))
							.withVariance(weightsLoader.getBatchNormLayerVariance(
									"batch_normalization_9_moving_variance0", 64))
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withActivationFunction("relu_9", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
						.withConvolutionalAxons("conv2d_10")
							.withConnectionWeights(
									weightsLoader.getConvolutionalLayerWeights("conv2d_10_kernel0", 3, 3, 64, 96))
							.withFilterSize(3, 3).withFilterCount(96).withValidPadding()
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
							.withConnectionToNeurons(new Neurons3D(71, 71, 96, false))
						.withBatchNormAxons("batch_normalization_10")
							.withBiasUnit()
							.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_10_beta0", 96))
							.withMean(weightsLoader.getBatchNormLayerMean(
									"batch_normalization_10_moving_mean0", 96))
							.withVariance(weightsLoader.getBatchNormLayerVariance(
									"batch_normalization_10_moving_variance0",96))
							.withAxonsContextConfigurer(
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(71, 71, 96, false))
						.withActivationFunction("relu_10", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
					.endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT)
				.withParallelPaths()
					.withPath()
						.withConvolutionalAxons("conv2d_11")
						.withConnectionWeights(
								weightsLoader.getConvolutionalLayerWeights("conv2d_11_kernel0", 3, 3, 192, 192))
						.withStride(2, 2).withFilterSize(3, 3).withFilterCount(192).withValidPadding()
						.withAxonsContextConfigurer(
								c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(35, 35, 192, false))
						.withBatchNormAxons("batch_normalization_11").withBiasUnit()
						.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_11_beta0", 192))
						.withMean(weightsLoader.getBatchNormLayerMean(
								"batch_normalization_11_moving_mean0", 192))
						.withVariance(weightsLoader.getBatchNormLayerVariance(
								"batch_normalization_11_moving_variance0", 192))
						.withAxonsContextConfigurer(
								c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(35, 35, 192, false))
						.withActivationFunction("relu_11", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.endPath()
				.withPath()
					.withMaxPoolingAxons("max_pooling_2")
						.withFilterSize(3, 3)
						.withStride(2, 2)
						.withValidPadding()
					.withConnectionToNeurons(new Neurons3D(35, 35, 192, false))
				.endPath()
			.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
	}

	@Override
	public String getName() {
		return "inceptionv4_stem";
	}
}
