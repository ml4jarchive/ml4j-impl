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

package org.ml4j.nn.supervised;

import org.ml4j.nn.LayeredFeedForwardNeuralNetworkContext;
import org.ml4j.nn.components.DirectedComponentsContext;

/**
 * Simple default implementation of FeedForwardNeuralNetworkContext.
 * 
 * @author Michael Lavelle
 * 
 */
public class LayeredFeedForwardNeuralNetworkContextImpl extends FeedForwardNeuralNetworkContextImpl
		implements LayeredFeedForwardNeuralNetworkContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private int startLayerIndex;

	private Integer endLayerIndex;

	public LayeredFeedForwardNeuralNetworkContextImpl(DirectedComponentsContext directedComponentsContext, int startLayerIndex,
			Integer endLayerIndex, boolean isTrainingContext) {
		super(directedComponentsContext, isTrainingContext);
		this.startLayerIndex = startLayerIndex;
		this.endLayerIndex = endLayerIndex;

		if (endLayerIndex != null && startLayerIndex > endLayerIndex) {
			throw new IllegalArgumentException("Start layer index cannot be greater " + "than end layer index");
		}

	}

	@Override
	public int getStartLayerIndex() {
		return startLayerIndex;
	}

	@Override
	public Integer getEndLayerIndex() {
		return endLayerIndex;
	}

	@Override
	public LayeredFeedForwardNeuralNetworkContext asNonTrainingContext() {
		return new LayeredFeedForwardNeuralNetworkContextImpl(getDirectedComponentsContext().asNonTrainingContext(), startLayerIndex, endLayerIndex, false);
	}

	@Override
	public LayeredFeedForwardNeuralNetworkContext asTrainingContext() {
		return new LayeredFeedForwardNeuralNetworkContextImpl(getDirectedComponentsContext().asTrainingContext(), startLayerIndex, endLayerIndex, true);
	}

}
