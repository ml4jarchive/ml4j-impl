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

package org.ml4j.nn.synapses;

import org.ml4j.nn.components.DirectedComponentActivationContextImpl;
import org.ml4j.nn.components.DirectedComponentsContext;

/**
 * Simple default implementation of DirectedSynapsesContext.
 * 
 * @author Michael Lavelle
 * 
 */
public class DirectedSynapsesContextImpl extends DirectedComponentActivationContextImpl implements DirectedSynapsesContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * The MatrixFactory we configure for this context.
	 */
	private boolean withFreezeOut;

	/**
	 * Construct a new default DirectedSynapsesContext.
	 * 
	 * @param matrixFactory The MatrixFactory we configure for this context
	 * @param withFreezeOut Whether to freeze out these Synapses.
	 */
	public DirectedSynapsesContextImpl(DirectedComponentsContext directedComponentsContext, boolean withFreezeOut) {
		super(directedComponentsContext);
		this.withFreezeOut = withFreezeOut;

	}
	
	@Override
	public boolean isWithFreezeOut() {
		return withFreezeOut;
	}

	@Override
	public DirectedSynapsesContext withFreezeOut(boolean withFreezeOut) {
		this.withFreezeOut = withFreezeOut;
		return this;
	}
}
