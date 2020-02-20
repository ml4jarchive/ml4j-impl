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

import java.util.Optional;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.BatchNormConfig;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.WeightsFormat;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
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
	 * @param name					   The name of this layer.
	 * @param directedComponentFactory A factory implementation to create directed
	 *                                 components.
	 * @param primaryAxons             The primary Axons.
	 * @param activationFunction       The primary activation function.
	 * @param batchNormConfig          The batch norm config for this layer, or null if no batch norm.
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			ConvolutionalAxons primaryAxons, DifferentiableActivationFunction activationFunction, 
			BatchNormConfig<Neurons3D> batchNormConfig) {
		super(name, directedComponentFactory, primaryAxons, activationFunction, batchNormConfig);
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
	 * @param batchNormConfig          The batch norm config for this layer, or null if no batch norm.
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, Axons3DConfig axonsConfig, WeightsFormat weightsFormat,
			DifferentiableActivationFunction primaryActivationFunction,
			BatchNormConfig<Neurons3D> batchNormConfig) {
		super(name, directedComponentFactory,
				axonsFactory.createConvolutionalAxons(axonsConfig,
						new WeightsMatrixImpl(null, weightsFormat), null),
				primaryActivationFunction, batchNormConfig);
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
	 * @param batchNormConfig          The batch norm config for this layer, or null if no batch norm.
	 */
	public ConvolutionalFeedForwardLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, Axons3DConfig axonsConfig, WeightsMatrix weightsMatrix, BiasVector biasMatrix, 
			DifferentiableActivationFunction primaryActivationFunction,
			BatchNormConfig<Neurons3D> batchNormConfig) {
		super(name, directedComponentFactory,
				axonsFactory.createConvolutionalAxons(axonsConfig,
						weightsMatrix, biasMatrix),
				primaryActivationFunction, batchNormConfig);
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public ConvolutionalFeedForwardLayer dup(DirectedComponentFactory directedComponentFactory) {
		return new ConvolutionalFeedForwardLayerImpl(name,  // TODO -remove cast
				directedComponentFactory,  this.primaryAxons.dup(), this.primaryActivationFunction, (BatchNormConfig<Neurons3D>)batchNormConfig.dup());
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

	@Override
	public Optional<AxonsContext> getBatchNormAxonsContext(DirectedComponentsContext directedComponentsContext) {
		return Optional.empty();
	}

	@Override
	public AxonsContext getPrimaryAxonsContext(DirectedComponentsContext directedComponentsContext) {
		return getAxonsContext(directedComponentsContext, getPrimaryAxonsComponentName());
	}
}
