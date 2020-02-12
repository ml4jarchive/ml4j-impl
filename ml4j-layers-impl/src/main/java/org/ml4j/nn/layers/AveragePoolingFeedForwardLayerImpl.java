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
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

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
	
	/**
	 * 
	 * @param name The name of this average pooling layer.
	 * @param directedComponentFactory The directed component factory.
	 * @param primaryAxons The average pooling axons within this layer.
	 * @param primaryActivationFunction The activation function within this layer.s
	 * @param withBatchNorm Whether to enable batch norm.
	 */
	public AveragePoolingFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AveragePoolingAxons primaryAxons, DifferentiableActivationFunction primaryActivationFunction,
			boolean withBatchNorm) {
		super(name, directedComponentFactory, primaryAxons, primaryActivationFunction,
				withBatchNorm);
	}
	
	/**
	 * 
	 * @param name The name of this average pooling layer.
	 * @param directedComponentFactory The directed component factory.
	 * @param activationFunctionFactory The activation function factory.
	 * @param primaryAxons The average pooling axons within this layer.
	 * @param withBatchNorm Whether to enable batch norm.
	 */
	public AveragePoolingFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			DifferentiableActivationFunctionFactory activationFunctionFactory, AveragePoolingAxons primaryAxons, boolean withBatchNorm) {
		super(name, directedComponentFactory, primaryAxons, activationFunctionFactory.createLinearActivationFunction(), withBatchNorm);
	}
	
	/**
	 * 
	 * @param name The name of this average pooling layer.
	 * @param directedComponentFactory The directed component factory.
	 * @param axonsFactory The axons factory.
	 * @param activationFunctionFactory The activation function factory.
	 * @param config The axons config.
	 * @param withBatchNorm	Whether to enable batch norm.
	 */
	public AveragePoolingFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, DifferentiableActivationFunctionFactory activationFunctionFactory, Axons3DConfig config, boolean withBatchNorm) {
		super(name, directedComponentFactory,
				axonsFactory.createAveragePoolingAxons(config),
				activationFunctionFactory.createLinearActivationFunction(), withBatchNorm);
	}
	

	@Override
	public AveragePoolingFeedForwardLayer dup(DirectedComponentFactory directedComponentFactory) {
		return new AveragePoolingFeedForwardLayerImpl(name, directedComponentFactory,
				primaryAxons.dup(), this.primaryActivationFunction, withBatchNorm);
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
