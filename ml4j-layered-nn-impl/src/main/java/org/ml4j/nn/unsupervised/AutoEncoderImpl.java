/*
 * Copyright 2017 the original author or authors.
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

package org.ml4j.nn.unsupervised;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.CostAndGradientsImpl;
import org.ml4j.nn.LayeredFeedForwardNeuralNetworkBase;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.layers.DirectedLayerChain;
import org.ml4j.nn.layers.DirectedLayerChainImpl;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of AutoEncoder consisting of FeedForwardLayers.
 *
 * @author Michael Lavelle
 */
public class AutoEncoderImpl extends LayeredFeedForwardNeuralNetworkBase<AutoEncoderContext, AutoEncoder>
		implements AutoEncoder {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(AutoEncoderImpl.class);

	private DirectedComponentFactory directedComponentFactory;

	/**
	 * Constructor for a simple 2-layer AutoEncoder.
	 * 
	 * @param encodingLayer The encoding Layer
	 * @param decodingLayer The decoding Layer
	 */
	public AutoEncoderImpl(DirectedComponentFactory directedComponentFactory, FeedForwardLayer<?, ?> encodingLayer,
			FeedForwardLayer<?, ?> decodingLayer) {
		this(directedComponentFactory, new DirectedLayerChainImpl<>(Arrays.asList(encodingLayer, decodingLayer)));
	}

	/**
	 * Constructor for a multi-layer AutoEncoder.
	 * 
	 * @param layers The layers
	 */
	public AutoEncoderImpl(DirectedComponentFactory directedComponentFactory, List<FeedForwardLayer<?, ?>> layers) {
		this(directedComponentFactory, new DirectedLayerChainImpl<>(layers));
	}

	protected AutoEncoderImpl(DirectedComponentFactory directedComponentFactory,
			DirectedLayerChain<FeedForwardLayer<?, ?>> initialisingComponentChain) {
		super(directedComponentFactory, initialisingComponentChain);
	}

	@Override
	public AutoEncoder dup() {
		return new AutoEncoderImpl(directedComponentFactory, this.initialisingComponentChain);
	}

	@Override
	public NeuronsActivation encode(NeuronsActivation unencoded, AutoEncoderContext context) {
		LOGGER.debug("Encoding through AutoEncoder");
		if (context.getEndLayerIndex() == null || context.getEndLayerIndex() >= (getNumberOfLayers() - 1)) {
			throw new IllegalArgumentException("End layer index for encoding through AutoEncoder "
					+ " must be specified and must not be the index of the last layer");
		}
		return forwardPropagate(unencoded, context).getOutput();
	}

	@Override
	public NeuronsActivation decode(NeuronsActivation encoded, AutoEncoderContext context) {
		LOGGER.debug("Decoding through AutoEncoder");
		if (context.getStartLayerIndex() == 0) {
			throw new IllegalArgumentException("Start layer index for decoding through AutoEncoder "
					+ " must not be 0 - the index of the first layer");
		}
		return forwardPropagate(encoded, context).getOutput();
	}

	@Override
	public void train(NeuronsActivation inputActivations, AutoEncoderContext trainingContext) {
		if (inputActivations.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException(
					"Only neurons actiavation with ROWS_SPAN_FEATURE_SET " + "orientation supported currently");
		}

		super.train(inputActivations, inputActivations, trainingContext);
	}

	@Override
	public CostAndGradientsImpl getCostAndGradients(NeuronsActivation inputActivations,
			AutoEncoderContext trainingContext) {
		return super.getCostAndGradients(inputActivations, inputActivations, trainingContext);
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return trailingActivationFunctionComponentChain.decompose();
	}

	@Override
	public AutoEncoderContext getContext(DirectedComponentsContext context, int componentIndex) {
		throw new UnsupportedOperationException();

	}

	@Override
	public Neurons getInputNeurons() {
		// TODO Auto-generated method stub
		return trailingActivationFunctionComponentChain.getInputNeurons();
	}

	@Override
	public Neurons getOutputNeurons() {
		return trailingActivationFunctionComponentChain.getOutputNeurons();
	}

}
