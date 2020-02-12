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
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.WeightsFormat;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

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
	 * @param name					   The name of this layer.
	 * @param directedComponentFactory A factory implementation to create directed
	 *                                 components.
	 * @param primaryAxons             The primary Axons.
	 * @param activationFunction       The primary activation function.
	 * @param withBatchNorm            Whether to enable batch norm for this Layer.
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			ConvolutionalAxons primaryAxons, DifferentiableActivationFunction activationFunction, boolean withBatchNorm) {
		super(name, directedComponentFactory, primaryAxons, activationFunction, withBatchNorm);
	}
	
	/**
	 * 
	 * @param name	The name of the layer.
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param axonsConfig				The config of the convolutional axons.
	 * @param weightsFormat				The weights format
	 * @param primaryActivationFunction The primary activation function of this layer.
	 * @param withBatchNorm				Whether to enable batch norm.s
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, Axons3DConfig axonsConfig, WeightsFormat weightsFormat,
			DifferentiableActivationFunction primaryActivationFunction,
			boolean withBatchNorm) {
		super(name, directedComponentFactory,
				axonsFactory.createConvolutionalAxons(axonsConfig,
						new WeightsMatrixImpl(null, weightsFormat), null),
				primaryActivationFunction, withBatchNorm);
	}
	
	/**
	 * 
	 * @param name	The name of the layer.
	 * @param directedComponentFactory  A factory implementation to create directed
	 *                                  components.	
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param axonsConfig				The config of the convolutional axons.
	 * @param weightsMatrix				The weights matrix
	 * @param biasMatrix				The bias matrix - only required if the left neurons of the axons config have a bias unit
	 * 									specified - may be null otherwise.
	 * @param primaryActivationFunction The primary activation function of this layer.
	 * @param withBatchNorm				Whether to enable batch norm.s
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, Axons3DConfig axonsConfig, WeightsMatrix weightsMatrix, BiasMatrix biasMatrix, 
			DifferentiableActivationFunction primaryActivationFunction,
			boolean withBatchNorm) {
		super(name, directedComponentFactory,
				axonsFactory.createConvolutionalAxons(axonsConfig,
						weightsMatrix, biasMatrix),
				primaryActivationFunction, withBatchNorm);
	}
	
	@Override
	public ConvolutionalFeedForwardLayer dup(DirectedComponentFactory directedComponentFactory) {
		return new ConvolutionalFeedForwardLayerImpl(name, 
				directedComponentFactory,  this.primaryAxons.dup(), this.primaryActivationFunction, withBatchNorm);
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

	@Override
	public int getZeroPadding() {
		return getPrimaryAxons().getConfig().getPaddingWidth();
	}
}
