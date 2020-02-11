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
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Default implementation of an AveragePoolingFeedForwardLayer.
 * 
 * @author Michael Lavelle
 */
public class AveragePoolingFeedForwardLayerImpl
		extends FeedForwardLayerBase<AveragePoolingAxons, AveragePoolingFeedForwardLayer>
		implements AveragePoolingFeedForwardLayer {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private DifferentiableActivationFunctionFactory activationFunctionFactory;

	/**
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param activationFunctionFactory A factory implementation to create
	 *                                  activation functions.
	 * @param primaryAxons              The average pooling Axons.
	 * @param matrixFactory             The matrix factory.
	 * @param withBatchNorm             Whether to enable batch norm
	 */
	public AveragePoolingFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			DifferentiableActivationFunctionFactory activationFunctionFactory, AveragePoolingAxons primaryAxons,
			MatrixFactory matrixFactory, boolean withBatchNorm) {
		super(name, directedComponentFactory, primaryAxons, activationFunctionFactory.createLinearActivationFunction(),
				matrixFactory, withBatchNorm);
		this.activationFunctionFactory = activationFunctionFactory;
	}

	/**
	 * 
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param activationFunctionFactory A factory implementation to create
	 *                                  activation functions.
	 * @param inputNeurons              The input Neurons.
	 * @param outputNeurons             The output Neurons
	 * @param matrixFactory             The MatrixFactory to use to initialise the
	 *                                  weights
	 * @param withBatchNorm             Whether to enable batch norm
	 */
	public AveragePoolingFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, DifferentiableActivationFunctionFactory activationFunctionFactory,
			Neurons3D inputNeurons, Neurons3D outputNeurons, MatrixFactory matrixFactory, boolean withBatchNorm) {
		super(name, directedComponentFactory,
				axonsFactory.createAveragePoolingAxons(inputNeurons, outputNeurons, new Axons3DConfig()),
				activationFunctionFactory.createLinearActivationFunction(), matrixFactory, withBatchNorm);
	}

	@Override
	public AveragePoolingFeedForwardLayer dup() {
		return new AveragePoolingFeedForwardLayerImpl(name, directedComponentFactory, activationFunctionFactory,
				primaryAxons.dup(), matrixFactory, withBatchNorm);
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
}
