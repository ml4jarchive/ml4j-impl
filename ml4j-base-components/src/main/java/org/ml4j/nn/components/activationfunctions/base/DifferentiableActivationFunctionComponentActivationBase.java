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
package org.ml4j.nn.components.activationfunctions.base;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.base.DefaultChainableDirectedComponentActivationBase;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for implementations of DifferentiableActivationFunctionActivation.
 * 
 * Encapsulates the activations from an activation through a DifferentiableActivationFunctionComponent
 * 
 * @author Michael Lavelle
 */
public abstract class DifferentiableActivationFunctionComponentActivationBase<L extends DifferentiableActivationFunctionComponent> extends DefaultChainableDirectedComponentActivationBase<L> implements DifferentiableActivationFunctionComponentActivation {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DifferentiableActivationFunctionComponentActivationBase.class);
		
	protected NeuronsActivation input;
	
	/**
	 * @param input The input to the activation function component
	 * @param output The output from the activation function component
	 */
	public DifferentiableActivationFunctionComponentActivationBase(L activationFunctionComponent, NeuronsActivation input, NeuronsActivation output) {
		super(activationFunctionComponent, output);
		this.input = input;
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}
}
