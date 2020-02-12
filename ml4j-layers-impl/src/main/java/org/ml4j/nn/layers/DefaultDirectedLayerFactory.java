/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.layers;

import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsConfig;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons;

/**
 * Default implementation of DirectedLayerFactory.
 * 
 * @author Michael Lavelle
 */
public class DefaultDirectedLayerFactory implements DirectedLayerFactory {

	protected AxonsFactory axonsFactory;
	protected DirectedComponentFactory directedComponentFactory;
	protected DifferentiableActivationFunctionFactory differentiableActivationFunctionFactory;

	/**
	 * @param axonsFactory The axons factory.
	 * @param differentiableActivationFunctionFactory The activation function factory.
	 * @param directedComponentFactory The directed component factory.
	 */
	public DefaultDirectedLayerFactory(AxonsFactory axonsFactory,
			DifferentiableActivationFunctionFactory differentiableActivationFunctionFactory,
			DirectedComponentFactory directedComponentFactory) {
		this.axonsFactory = axonsFactory;
		this.differentiableActivationFunctionFactory = differentiableActivationFunctionFactory;
		this.directedComponentFactory = directedComponentFactory;
	}

	@Override
	public FullyConnectedFeedForwardLayer createFullyConnectedFeedForwardLayer(String name, AxonsConfig<Neurons, Neurons> axonsConfig, WeightsMatrix connectionWeights, BiasMatrix biases,
			ActivationFunctionType activationFunctionType, ActivationFunctionProperties activationFunctionProperties,
			boolean withBatchNorm) {
		return new FullyConnectedFeedForwardLayerImpl(name, directedComponentFactory, axonsFactory,
				axonsConfig, connectionWeights, biases, differentiableActivationFunctionFactory.createActivationFunction(activationFunctionType,
						activationFunctionProperties),
				withBatchNorm);
	}

	@Override
	public MaxPoolingFeedForwardLayer createMaxPoolingFeedForwardLayer(String name, Axons3DConfig axonsConfig, boolean scaleOutputs) {
		return new MaxPoolingFeedForwardLayerImpl(name, directedComponentFactory, axonsFactory, axonsConfig,
				differentiableActivationFunctionFactory, scaleOutputs, false);
	}

	@Override
	public AveragePoolingFeedForwardLayer createAveragePoolingFeedForwardLayer(String name, Axons3DConfig axonsConfig) {
		return new AveragePoolingFeedForwardLayerImpl(name, directedComponentFactory, axonsFactory,
				differentiableActivationFunctionFactory, axonsConfig,  false);
	}

	@Override
	public ConvolutionalFeedForwardLayer createConvolutionalFeedForwardLayer(String name, Axons3DConfig axons3DConfig,  WeightsMatrix connectionWeights, BiasMatrix biases,
			ActivationFunctionType activationFunctionType, ActivationFunctionProperties activationFunctionProperties,
			boolean withBatchNorm) {
		return new ConvolutionalFeedForwardLayerImpl(name, directedComponentFactory, axonsFactory, axons3DConfig, connectionWeights, biases, differentiableActivationFunctionFactory
						.createActivationFunction(activationFunctionType, activationFunctionProperties), withBatchNorm);
	}

}
