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
import java.util.Optional;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.AxonsConfig;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.BatchNormConfig;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
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
	 * @param name					   The name of this layer.
	 * @param directedComponentFactory A factory implementation to create directed
	 *                                 components.
	 * @param primaryAxons             The primary Axons
	 * @param activationFunction       The primary activation function.
	 * @param batchNormConfig          The batch norm config for this layer, or null if no batch norm.
	 */
	public FullyConnectedFeedForwardLayerImpl(String name,DirectedComponentFactory directedComponentFactory,
			FullyConnectedAxons primaryAxons, DifferentiableActivationFunction activationFunction, BatchNormConfig<?> batchNormConfig) {
		super(name, directedComponentFactory, primaryAxons, activationFunction, batchNormConfig);
	}

	/**
	 * 
	 * @param name					    The name of this layer.
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param axonsConfig               The input/output neurons config for the axons of this layer.
	 * @param primaryActivationFunction The primary activation function.
	 * @param batchNormConfig          The batch norm config for this layer, or null if no batch norm.
	 */
	public FullyConnectedFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, AxonsConfig<Neurons, Neurons> axonsConfig, 
			DifferentiableActivationFunction primaryActivationFunction, 
			BatchNormConfig<?> batchNormConfig) {
		super(name, directedComponentFactory, axonsFactory.createFullyConnectedAxons(axonsConfig, 
				new WeightsMatrixImpl(null, 
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE),
						WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)) , null),
				primaryActivationFunction, batchNormConfig);
	}

	/**
	 * 
	 * @param name					    The name of this layer.
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param axonsConfig               The input/output neurons config for the axons of this layer.
	 * @param connectionWeights			The connection weights of this layer.
	 * @param biases					The biases for this layer - only required if the axons config has a left neurons
	 * 									with bias unit - may be null otherwise.
	 * @param primaryActivationFunction The primary activation function.
	 * @param batchNormConfig          The batch norm config for this layer, or null if no batch norm.
	 */
	public FullyConnectedFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, AxonsConfig<Neurons, Neurons> axonsConfig, 
			WeightsMatrix connectionWeights, BiasVector biases, DifferentiableActivationFunction primaryActivationFunction, BatchNormConfig<?> batchNormConfig) {
		super(name, directedComponentFactory,
				axonsFactory.createFullyConnectedAxons(axonsConfig, 
						connectionWeights,
						biases),
				primaryActivationFunction, batchNormConfig);
	}

	@Override
	public FullyConnectedFeedForwardLayer dup(DirectedComponentFactory directedComponentFactory) {
		return new FullyConnectedFeedForwardLayerImpl(name, directedComponentFactory, 
				this.primaryAxons.dup(), primaryActivationFunction, batchNormConfig.dup());
	}

	@Override
	public Optional<AxonsContext> getBatchNormAxonsContext(DirectedComponentsContext arg0) {
		return Optional.empty();
	}

	@Override
	public AxonsContext getPrimaryAxonsContext(DirectedComponentsContext directedComponentsContext) {
		return getAxonsContext(directedComponentsContext, this.getPrimaryAxonsComponentName());
	}
}
