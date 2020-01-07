/*
 * Copyright 2019 the original author or authors.
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
package org.ml4j.nn.components.base;

import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for implementations of DefaultChainableDirectedComponentActivation.
 * 
 * Encapsulates the activations from a forward propagation through a ChainableDirectedComponent
 * 
 * 
 * @author Michael Lavelle
 * 
 * @param <L> The specific type of DefaultChainableDirectedComponent from which this activation originated.
 */
public abstract class DefaultChainableDirectedComponentActivationBase<L extends DefaultChainableDirectedComponent<?, ?>> implements DefaultChainableDirectedComponentActivation {
	
	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultChainableDirectedComponentActivationBase.class);
	
	/**
	 * The NeuronsActivation output on the RHS of the forward propagation.
	 */
	protected NeuronsActivation output;
	
	/**
	 * The component from which this activation originated.
	 */
	protected L originatingComponent;
	
	/**
	 * @param originatingComponent The component from which this activation originated
	 * @param output The NeuronsActivation output on the RHS of the forward propagation.
	 */
	public DefaultChainableDirectedComponentActivationBase(L originatingComponent, NeuronsActivation output) {
		this.output = output;
		this.originatingComponent = originatingComponent;
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}
}


