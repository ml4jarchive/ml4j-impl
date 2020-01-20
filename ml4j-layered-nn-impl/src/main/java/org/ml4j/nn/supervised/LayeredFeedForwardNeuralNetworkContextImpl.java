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

import java.util.HashMap;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.LayeredFeedForwardNeuralNetworkContext;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.DirectedLayerContextImpl;

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

	private HashMap<Integer, DirectedLayerContext> directedLayerContexts;

	private boolean isTrainingContext;

	/**
	 * 
	 * @param matrixFactory
	 * @param startLayerIndex
	 * @param endLayerIndex
	 */
	public LayeredFeedForwardNeuralNetworkContextImpl(MatrixFactory matrixFactory, int startLayerIndex,
			Integer endLayerIndex, boolean isTrainingContext) {
		super(matrixFactory, isTrainingContext);
		this.isTrainingContext = isTrainingContext;
		this.startLayerIndex = startLayerIndex;
		this.endLayerIndex = endLayerIndex;

		if (endLayerIndex != null && startLayerIndex > endLayerIndex) {
			throw new IllegalArgumentException("Start layer index cannot be greater " + "than end layer index");
		}

		this.directedLayerContexts = new HashMap<>();
	}

	@Override
	public DirectedLayerContext getLayerContext(int layerIndex) {

		DirectedLayerContext layerContext = directedLayerContexts.get(layerIndex);
		if (layerContext == null) {
			layerContext = new DirectedLayerContextImpl(layerIndex, getMatrixFactory(), isTrainingContext);
			directedLayerContexts.put(layerIndex, layerContext);
		}

		return layerContext;
	}

	public boolean isTrainingContext() {
		return isTrainingContext;
	}

	@Override
	public int getStartLayerIndex() {
		return startLayerIndex;
	}

	@Override
	public Integer getEndLayerIndex() {
		return endLayerIndex;
	}
}
