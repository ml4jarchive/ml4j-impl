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

import java.util.HashMap;
import java.util.Map;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.synapses.DirectedSynapsesContext;
import org.ml4j.nn.synapses.DirectedSynapsesContextImpl;

/**
 * Simple default implementation of DirectedLayerContext.
 * 
 * @author Michael Lavelle
 * 
 */
public class DirectedLayerContextImpl implements DirectedLayerContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * The MatrixFactory we configure for this context.
	 */
	private MatrixFactory matrixFactory;

	private int layerIndex;

	private boolean withFreezeOut;
	private boolean isTrainingContext;
	private Map<Integer, DirectedSynapsesContext> synapsesContextsBySynapsesIndex;

	/**
	 * Construct a new DirectedLayerContext.
	 * 
	 * @param layerIndex    The index of the layer
	 * @param matrixFactory The MatrixFactory we configure for this context
	 */
	public DirectedLayerContextImpl(int layerIndex, MatrixFactory matrixFactory, boolean isTrainingContext) {
		this.matrixFactory = matrixFactory;
		this.layerIndex = layerIndex;
		this.synapsesContextsBySynapsesIndex = new HashMap<>();
		this.isTrainingContext = isTrainingContext;
	}

	@Override
	public MatrixFactory getMatrixFactory() {
		return matrixFactory;
	}

	@Override
	public DirectedSynapsesContext getSynapsesContext(int synapsesIndex) {

		DirectedSynapsesContext synapsesContext = synapsesContextsBySynapsesIndex.get(synapsesIndex);
		if (synapsesContext == null) {
			synapsesContext = new DirectedSynapsesContextImpl(matrixFactory, isTrainingContext, withFreezeOut);

		}
		if (synapsesContext.isWithFreezeOut() != withFreezeOut) {
			synapsesContext.setWithFreezeOut(withFreezeOut);
			synapsesContextsBySynapsesIndex.put(synapsesIndex, synapsesContext);
		}

		return synapsesContext;
	}

	public boolean isWithFreezeOut() {
		return withFreezeOut;
	}

	public void setWithFreezeOut(boolean withFreezeOut) {
		this.withFreezeOut = withFreezeOut;
	}

	@Override
	public String toString() {
		return "DirectedLayerContextImpl [layerIndex=" + layerIndex + "]";
	}

	@Override
	public boolean isTrainingContext() {
		return isTrainingContext;
	}
}
