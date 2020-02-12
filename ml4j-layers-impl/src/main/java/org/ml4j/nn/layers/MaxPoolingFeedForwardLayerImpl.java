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

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.MaxPoolingAxons;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

/**
 * Default implementation of a MaxPoolingFeedForwardLayer.
 * 
 * @author Michael Lavelle
 */
public class MaxPoolingFeedForwardLayerImpl extends FeedForwardLayerBase<MaxPoolingAxons, MaxPoolingFeedForwardLayer>
		implements MaxPoolingFeedForwardLayer {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * 
	 * @param name	The name of the max pooling layer.
	 * @param directedComponentFactory The directed component factory.
	 * @param axonsFactory	The axons factory.
	 * @param axons3DConfig The axons config.
	 * @param activationFunctionFactory The activation function factory.
	 * @param scaleOutputs	Whether to scale the outputs by the pooling factory.
	 * @param withBatchNorm	Whether to enable batch norm.
	 */
	public MaxPoolingFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory, AxonsFactory axonsFactory,
			Axons3DConfig axons3DConfig, 
			DifferentiableActivationFunctionFactory activationFunctionFactory, boolean scaleOutputs, boolean withBatchNorm) {
		super(name, directedComponentFactory,
				axonsFactory.createMaxPoolingAxons(axons3DConfig, scaleOutputs),
				activationFunctionFactory.createLinearActivationFunction(), withBatchNorm);
	}
	
	/**
	 * @param name	The name of the max pooling layer.
	 * @param directedComponentFactory The directed component factory.
	 * @param primaryAxons The max pooling axons within this layer.
	 * @param activationFunction The activation function within this layer.
	 * @param withBatchNorm Whether to enable batch norm.
	 */
	public MaxPoolingFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			MaxPoolingAxons primaryAxons, DifferentiableActivationFunction activationFunction, boolean withBatchNorm) {
		super(name, directedComponentFactory, primaryAxons, activationFunction, withBatchNorm);
	}

	@Override
	public MaxPoolingFeedForwardLayer dup(DirectedComponentFactory directedComponentFactory) {
		return new MaxPoolingFeedForwardLayerImpl(name, directedComponentFactory, primaryAxons.dup(), this.primaryActivationFunction, withBatchNorm);
	}

	@Override
	public int getFilterHeight() {
		return getPrimaryAxons().getConfig().getFilterHeight();
	}

	@Override
	public int getFilterWidth() {
		return getPrimaryAxons().getConfig().getFilterWidth();
	}

	@Override
	public int getStride() {
		return getPrimaryAxons().getConfig().getStrideWidth();

	}
}
