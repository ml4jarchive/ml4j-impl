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

package org.ml4j.nn;

import java.util.List;

import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.layers.DirectedLayerChain;
import org.ml4j.nn.layers.FeedForwardLayer;

/**
 * Default base implementation of a FeedForwardNeuralNetwork.
 *
 * @author Michael Lavelle
 */
public abstract class LayeredFeedForwardNeuralNetworkBase<C extends LayeredFeedForwardNeuralNetworkContext, N extends LayeredFeedForwardNeuralNetwork<C, N>>
		extends FeedForwardNeuralNetworkBase<C, DirectedLayerChain<FeedForwardLayer<?, ?>>, N>
		implements LayeredNeuralNetwork<FeedForwardLayer<?, ?>, C, N> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public LayeredFeedForwardNeuralNetworkBase(String name, DirectedComponentFactory directedComponentFactory,
			DirectedLayerChain<FeedForwardLayer<?, ?>> initialisingComponentChain) {
		super(name, directedComponentFactory, initialisingComponentChain);
	}

	@Override
	public List<FeedForwardLayer<?, ?>> getLayers() {
		return initialisingComponentChain.getComponents();
	}

	@Override
	public int getNumberOfLayers() {
		return initialisingComponentChain.getComponents().size();
	}

	@Override
	public FeedForwardLayer<?, ?> getLayer(int layerIndex) {
		return initialisingComponentChain.getComponents().get(layerIndex);
	}

	@Override
	public FeedForwardLayer<?, ?> getFirstLayer() {
		return initialisingComponentChain.getComponents().get(0);
	}

	@Override
	public FeedForwardLayer<?, ?> getFinalLayer() {
		return initialisingComponentChain.getComponents().get(getNumberOfLayers() - 1);
	}

}
