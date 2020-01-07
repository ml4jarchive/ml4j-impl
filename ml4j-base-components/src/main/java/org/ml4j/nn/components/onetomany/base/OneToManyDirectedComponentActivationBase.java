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
package org.ml4j.nn.components.onetomany.base;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base implementation for a OneToManyDirectedComponentActivation, encapsulating the activation from a OneToManyDirectedComponent.
 * 
 * @author Michael Lavelle
 */
public abstract class OneToManyDirectedComponentActivationBase implements OneToManyDirectedComponentActivation {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(OneToManyDirectedComponentActivationBase.class);
	
	private NeuronsActivation input;
	private int outputNeuronsActivationCount;
	
	/**
	 * OneToManyDirectedComponentActivationBase constructor
	 * 
	 * @param input The neurons activation input to the one to many component.
	 * @param outputNeuronsActivationCount The desired number of instances of output neuron activations, one for each of the components
	 * on the RHS of the OneToManyDirectedComponentActivation.
	 */
	public OneToManyDirectedComponentActivationBase(NeuronsActivation input, int outputNeuronsActivationCount) {
		this.input = input;
		this.outputNeuronsActivationCount = outputNeuronsActivationCount;
	}

	@Override
	public List<NeuronsActivation> getOutput() {
		List<NeuronsActivation> outputs = new ArrayList<>();
		for (int i = 0; i < outputNeuronsActivationCount; i++) {
			outputs.add(input);
		}
		return outputs;
	}
}
