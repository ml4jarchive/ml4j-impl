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
package org.ml4j.nn.activationfunctions.base;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
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
public abstract class DifferentiableActivationFunctionActivationBase implements DifferentiableActivationFunctionActivation {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DifferentiableActivationFunctionActivationBase.class);
		
	protected NeuronsActivation input;
	protected NeuronsActivation output;
	protected DifferentiableActivationFunction activationFunction;
	
	/**
	 * @param activationFunction The DifferentiableActivationFunction instance from which this activation was generated.
	 * @param input The input to the activation function component.
	 * @param output The output from the activation function component
	 */
	public DifferentiableActivationFunctionActivationBase(DifferentiableActivationFunction activationFunction, NeuronsActivation input, NeuronsActivation output) {
		this.input = input;
		this.output = output;
		this.activationFunction = activationFunction;
	}

	@Override
	public NeuronsActivation getInput() {
		return input;
	}
	
	@Override
	public NeuronsActivation getOutput() {
		return output;
	}
	
	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}
}
