/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.layers;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Default implementation of a ConvolutionalFeedForwardLayer.
 * 
 * @author Michael Lavelle
 */
public class ConvolutionalFeedForwardLayerImpl
		extends FeedForwardLayerBase<ConvolutionalAxons, ConvolutionalFeedForwardLayer>
		implements ConvolutionalFeedForwardLayer {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * @param directedComponentFactory A factory implementation to create directed
	 *                                 components.
	 * @param primaryAxons             The primary Axons.
	 * @param activationFunction       The primary activation function.
	 * @param matrixFactory            The matrix factory.
	 * @param withBatchNorm            Whether to enable batch norm for this Layer.
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			ConvolutionalAxons primaryAxons, DifferentiableActivationFunction activationFunction,
			MatrixFactory matrixFactory, boolean withBatchNorm) {
		super(name, directedComponentFactory, primaryAxons, activationFunction, matrixFactory, withBatchNorm);
	}

	/**
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param inputNeurons              The input Neurons.
	 * @param outputNeurons             The output Neurons
	 * @param stride                    The stride.
	 * @param zeroPadding               The amount of zero padding.
	 * @param primaryActivationFunction The primary activation function.
	 * @param matrixFactory             The MatrixFactory to use to initialise the
	 *                                  weights
	 * @param withBatchNorm             Whether to enable batch norm for this Layer.
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, Neurons3D inputNeurons, Neurons3D outputNeurons, int stride, int zeroPadding,
			DifferentiableActivationFunction primaryActivationFunction, MatrixFactory matrixFactory,
			boolean withBatchNorm) {
		super(name, directedComponentFactory,
				axonsFactory.createConvolutionalAxons(inputNeurons, outputNeurons,
						new Axons3DConfig().withStrideWidth(stride).withStrideHeight(stride)
								.withPaddingWidth(zeroPadding).withPaddingHeight(zeroPadding),
						null, null),
				primaryActivationFunction, matrixFactory, withBatchNorm);
	}

	/**
	 * Constructor for use with stride of 1 and no padding.
	 * 
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param inputNeurons              The input Neurons.
	 * @param outputNeurons             The output Neurons
	 * @param primaryActivationFunction The primary activation function.
	 * @param matrixFactory             The MatrixFactory to use to initialise the
	 *                                  weights
	 * @param connectionWeights         The connectionWeights
	 * @param withBatchNorm             Whether to enable batch norm for this Layer.
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, Neurons3D inputNeurons, Neurons3D outputNeurons,
			DifferentiableActivationFunction primaryActivationFunction, MatrixFactory matrixFactory,
			WeightsMatrix connectionWeights, BiasMatrix biases, boolean withBatchNorm) {
		this(name, directedComponentFactory, axonsFactory, inputNeurons, outputNeurons, 1, 0, primaryActivationFunction,
				matrixFactory, connectionWeights, biases, withBatchNorm);
	}

	/**
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param inputNeurons              The input Neurons.
	 * @param outputNeurons             The output Neurons
	 * @param stride                    The stride.
	 * @param zeroPadding               The amount of zero padding.
	 * @param primaryActivationFunction The primary activation function.
	 * @param matrixFactory             The MatrixFactory to use to initialise the
	 *                                  weights
	 * @param connectionWeights         The connectionWeights
	 * @param withBatchNorm             Whether to enable batch norm for this Layer.
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, Neurons3D inputNeurons, Neurons3D outputNeurons, int stride, int zeroPadding,
			DifferentiableActivationFunction primaryActivationFunction, MatrixFactory matrixFactory,
			WeightsMatrix connectionWeights, BiasMatrix biases, boolean withBatchNorm) {
		super(name, directedComponentFactory, axonsFactory.createConvolutionalAxons(inputNeurons, outputNeurons,
				new Axons3DConfig().withStrideWidth(stride).withStrideHeight(stride).withPaddingWidth(zeroPadding)
						.withPaddingHeight(zeroPadding),
				connectionWeights, biases), primaryActivationFunction, matrixFactory, withBatchNorm);
	}

	@Override
	public ConvolutionalFeedForwardLayer dup() {
		return new ConvolutionalFeedForwardLayerImpl(name, directedComponentFactory, primaryAxons.dup(),
				primaryActivationFunction, matrixFactory, withBatchNorm);
	}

	@Override
	public int getFilterHeight() {
		return getPrimaryAxons().getConfig().getFilterHeight(primaryAxons.getLeftNeurons(), primaryAxons.getRightNeurons());
	}

	@Override
	public int getFilterWidth() {
		return getPrimaryAxons().getConfig().getFilterWidth(primaryAxons.getLeftNeurons(), primaryAxons.getRightNeurons());
	}

	@Override
	public int getStride() {
		return getPrimaryAxons().getConfig().getStrideWidth();

	}

	@Override
	public int getZeroPadding() {
		return getPrimaryAxons().getConfig().getPaddingWidth();
	}
}
