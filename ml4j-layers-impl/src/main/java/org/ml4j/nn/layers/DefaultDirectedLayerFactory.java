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
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.BatchNormAxonsConfig;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.ConvolutionalAxonsConfig;
import org.ml4j.nn.axons.FullyConnectedAxonsConfig;
import org.ml4j.nn.axons.PoolingAxonsConfig;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Default implementation of DirectedLayerFactory.
 * 
 * @author Michael Lavelle
 */
public class DefaultDirectedLayerFactory implements DirectedLayerFactory {

	protected AxonsFactory axonsFactory;
	protected DirectedComponentFactory directedComponentFactory;
	protected DifferentiableActivationFunctionFactory differentiableActivationFunctionFactory;
	private DirectedComponentsContext directedComponentsContext;

	/**
	 * @param axonsFactory The axons factory.
	 * @param differentiableActivationFunctionFactory The activation function factory.
	 * @param directedComponentFactory The directed component factory.
	 * @param directedComponentsContext The directed components context.
	 */
	public DefaultDirectedLayerFactory(AxonsFactory axonsFactory,
			DifferentiableActivationFunctionFactory differentiableActivationFunctionFactory,
			DirectedComponentFactory directedComponentFactory,
			DirectedComponentsContext directedComponentsContext) {
		this.axonsFactory = axonsFactory;
		this.differentiableActivationFunctionFactory = differentiableActivationFunctionFactory;
		this.directedComponentFactory = directedComponentFactory;
		this.directedComponentsContext = directedComponentsContext;
	}

	@Override
	public FullyConnectedFeedForwardLayer createFullyConnectedFeedForwardLayer(String name, FullyConnectedAxonsConfig primaryAxonsConfig, WeightsMatrix connectionWeights, BiasVector biases,
			ActivationFunctionType activationFunctionType, ActivationFunctionProperties activationFunctionProperties,
			BatchNormAxonsConfig<Neurons> batchNormConfig) {
		FullyConnectedFeedForwardLayer layer = new FullyConnectedFeedForwardLayerImpl(name, directedComponentFactory, axonsFactory,
				primaryAxonsConfig.getAxonsConfig(), connectionWeights, biases, differentiableActivationFunctionFactory.createActivationFunction(activationFunctionType,
						activationFunctionProperties),
				batchNormConfig == null ? null : batchNormConfig.getBatchNormConfig());
		if (primaryAxonsConfig.getAxonsContextConfigurer() != null) {
			AxonsContext axonsContext = layer.getPrimaryAxonsContext(directedComponentsContext);
			primaryAxonsConfig.getAxonsContextConfigurer() .accept(axonsContext);
		}
		return layer;
	}

	@Override
	public MaxPoolingFeedForwardLayer createMaxPoolingFeedForwardLayer(String name, PoolingAxonsConfig axonsConfig, boolean scaleOutputs) {
		return new MaxPoolingFeedForwardLayerImpl(name, directedComponentFactory, axonsFactory, axonsConfig.getAxonsConfig(),
				differentiableActivationFunctionFactory, scaleOutputs);
	}

	@Override
	public AveragePoolingFeedForwardLayer createAveragePoolingFeedForwardLayer(String name, PoolingAxonsConfig axonsConfig) {
		return new AveragePoolingFeedForwardLayerImpl(name, directedComponentFactory, axonsFactory,
				differentiableActivationFunctionFactory, axonsConfig.getAxonsConfig());
	}

	@Override
	public ConvolutionalFeedForwardLayer createConvolutionalFeedForwardLayer(String name, ConvolutionalAxonsConfig primaryAxonsConfig,  WeightsMatrix connectionWeights, BiasVector biases,
			ActivationFunctionType activationFunctionType, ActivationFunctionProperties activationFunctionProperties,
			BatchNormAxonsConfig<Neurons3D> batchNormConfig) {
		ConvolutionalFeedForwardLayer layer =new ConvolutionalFeedForwardLayerImpl(name, directedComponentFactory, axonsFactory, primaryAxonsConfig.getAxonsConfig(), connectionWeights, biases, differentiableActivationFunctionFactory
						.createActivationFunction(activationFunctionType, activationFunctionProperties), batchNormConfig == null ? null : batchNormConfig.getBatchNormConfig());
		
		if (primaryAxonsConfig.getAxonsContextConfigurer() != null) {
			AxonsContext axonsContext = layer.getPrimaryAxonsContext(directedComponentsContext);
			primaryAxonsConfig.getAxonsContextConfigurer().accept(axonsContext);
		}
		return layer;
	}

}
