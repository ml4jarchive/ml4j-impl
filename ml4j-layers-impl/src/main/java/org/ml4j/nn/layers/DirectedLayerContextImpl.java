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

import java.util.Map;

import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentActivationContextImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.FreezeableNeuronsActivationContext;

/**
 * Simple default implementation of DirectedLayerContext.
 * 
 * @author Michael Lavelle
 * 
 */
public class DirectedLayerContextImpl extends DirectedComponentActivationContextImpl implements DirectedLayerContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private boolean withFreezeOut;
	private DirectedLayer<?, ?> directedLayer;

	/**
	 * Construct a new DirectedLayerContext.
	 * 
	 * @param layerIndex    The index of the layer
	 * @param matrixFactory The MatrixFactory we configure for this context
	 */
	public DirectedLayerContextImpl(DirectedLayer<?, ?> directedLayer, DirectedComponentsContext directedComponentsContext) {
		super(directedComponentsContext);
		this.directedLayer = directedLayer;
	}

	public boolean isWithFreezeOut() {
		return withFreezeOut;
	}

	@Override
	public DirectedLayerContext withFreezeOut(boolean withFreezeOut) {
		this.withFreezeOut = withFreezeOut;
		Map<String, AxonsContext> nestedAxonsContexts = directedLayer.getNestedContexts(getDirectedComponentsContext(), AxonsContext.class);
		nestedAxonsContexts.values().forEach(a -> a.addFreezeoutOverrideContext(this));
		return this;
	}

	@Override
	public String toString() {
		return "DirectedLayerContextImpl [withFreezeOut=" + withFreezeOut + "]";
	}

	@Override
	public String getOwningComponentName() {
		return directedLayer.getName();
	}

	@Override
	public void addFreezeoutOverrideContext(FreezeableNeuronsActivationContext<?> arg0) {
		throw new UnsupportedOperationException("Not currently supported");
	}

	@Override
	public void removeFreezeoutOverrideContext(FreezeableNeuronsActivationContext<?> arg0) {
		throw new UnsupportedOperationException("Not currently supported");
	}

	@Override
	public DirectedLayerContext asNonTrainingContext() {
		DirectedLayerContext layerContext = new DirectedLayerContextImpl(directedLayer, getDirectedComponentsContext().asNonTrainingContext());
		layerContext.withFreezeOut(isWithFreezeOut());
		return layerContext;
	}

	@Override
	public DirectedLayerContext asTrainingContext() {
		DirectedLayerContext layerContext = new DirectedLayerContextImpl(directedLayer, getDirectedComponentsContext().asTrainingContext());
		layerContext.withFreezeOut(isWithFreezeOut());
		return layerContext;
	}
}
