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

import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.FreezeableNeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationContextImpl;
import org.ml4j.nn.synapses.UndirectedSynapsesContext;
import org.ml4j.nn.synapses.UndirectedSynapsesContextImpl;

/**
 * Default implementation of UndirectedLayerContext.
 * 
 * @author Michael Lavelle
 */
public class UndirectedLayerContextImpl extends NeuronsActivationContextImpl implements UndirectedLayerContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private boolean withFreezeOut;
	
	private int layerIndex;
	
	private String layerName;

	/**
	 * @param layerIndex    The index of this Layer.
	 * @param matrixFactory The MatrixFactory.
	 */
	public UndirectedLayerContextImpl(String layerName, int layerIndex, MatrixFactory matrixFactory, boolean isTrainingContext) {
		super(matrixFactory, isTrainingContext);
		this.layerIndex = layerIndex;
		this.layerName = layerName;
	}
	
	@Override
	public UndirectedSynapsesContext createSynapsesContext(int synapsesIndex) {
		return new UndirectedSynapsesContextImpl(layerName, getMatrixFactory(), isTrainingContext(), withFreezeOut);
	}

	@Override
	public boolean isWithFreezeOut() {
		return withFreezeOut;
	}

	@Override
	public UndirectedLayerContext withFreezeOut(boolean withFreezeOut) {
		this.withFreezeOut = withFreezeOut;
		return this;
	}

	@Override
	public String toString() {
		return "UndirectedLayerContextImpl [layerIndex=" + layerIndex + "]";
	}

	@Override
	public String getOwningComponentName() {
		return layerName;
	}

	@Override
	public void addFreezeoutOverrideContext(FreezeableNeuronsActivationContext<?> arg0) {
		throw new UnsupportedOperationException("Not currently supported");
	}


	@Override
	public void removeFreezeoutOverrideContext(FreezeableNeuronsActivationContext<?> arg0) {
		throw new UnsupportedOperationException("Not currently supported");
	}
}
