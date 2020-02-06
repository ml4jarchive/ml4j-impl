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

import java.util.Arrays;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.format.features.Dimension;

/**
 * Default implementation of a FullyConnectedFeedForwardLayer.
 * 
 * @author Michael Lavelle
 */
public class FullyConnectedFeedForwardLayerImpl
		extends FeedForwardLayerBase<FullyConnectedAxons, FullyConnectedFeedForwardLayer>
		implements FullyConnectedFeedForwardLayer {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * @param directedComponentFactory A factory implementation to create directed
	 *                                 components.
	 * @param primaryAxons             The primary Axons
	 * @param activationFunction       The primary activation function.
	 * @param matrixFactory            The matrix factory.
	 * @param withBatchNorm            Whether to enable batch norm.
	 */
	public FullyConnectedFeedForwardLayerImpl(String name,DirectedComponentFactory directedComponentFactory,
			FullyConnectedAxons primaryAxons, DifferentiableActivationFunction activationFunction,
			MatrixFactory matrixFactory, boolean withBatchNorm) {
		super(name, directedComponentFactory, primaryAxons, activationFunction, matrixFactory, withBatchNorm);
	}

	/**
	 * 
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param inputNeurons              The input Neurons.
	 * @param outputNeurons             The output Neurons
	 * @param primaryActivationFunction The primary activation function.
	 * @param matrixFactory             The MatrixFactory to use to initialise the
	 *                                  weights
	 * @param withBatchNorm             Whether to enable batch norm.
	 */
	public FullyConnectedFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, Neurons inputNeurons, Neurons outputNeurons,
			DifferentiableActivationFunction primaryActivationFunction, MatrixFactory matrixFactory,
			boolean withBatchNorm) {
		super(name, directedComponentFactory, axonsFactory.createFullyConnectedAxons(inputNeurons, outputNeurons, 
				new WeightsMatrixImpl(null, 
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE),
						WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)) , null),
				primaryActivationFunction, matrixFactory, withBatchNorm);
	}

	/**
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param inputNeurons              The input Neurons.
	 * @param outputNeurons             The output Neurons
	 * @param primaryActivationFunction The primary activation function.
	 * @param matrixFactory             The MatrixFactory to use to initialise the
	 *                                  weights
	 * @param connectionWeights         The connection weights
	 * @param withBatchNorm             Whether to enable batch norm.
	 */
	public FullyConnectedFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, Neurons inputNeurons, Neurons outputNeurons,
			DifferentiableActivationFunction primaryActivationFunction, MatrixFactory matrixFactory,
			WeightsMatrix connectionWeights, BiasMatrix biases, boolean withBatchNorm) {
		super(name, directedComponentFactory,
				axonsFactory.createFullyConnectedAxons(inputNeurons, outputNeurons, 
						connectionWeights,
						biases),
				primaryActivationFunction, matrixFactory, withBatchNorm);
	}

	@Override
	public FullyConnectedFeedForwardLayer dup() {
		return new FullyConnectedFeedForwardLayerImpl(name, directedComponentFactory, primaryAxons.dup(),
				primaryActivationFunction, matrixFactory, withBatchNorm);
	}
}
