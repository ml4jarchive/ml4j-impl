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

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivationContext;

/**
 * 
 * 
 * @author Michael Lavelle
 */
public abstract class DifferentiableActivationFunctionComponentBase implements DifferentiableActivationFunctionComponent {

	/**
	 * Generated serialization id.
	 */
	private static final long serialVersionUID = -6033017517698579773L;
	
	protected DifferentiableActivationFunction activationFunction;
	protected Neurons neurons;
	
	public DifferentiableActivationFunctionComponentBase(Neurons neurons, DifferentiableActivationFunction activationFunction){
		this.activationFunction = activationFunction;
		this.neurons = neurons;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public NeuronsActivationContext getContext(DirectedComponentsContext context, int arg1) {
		return new NeuronsActivationContext() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public MatrixFactory getMatrixFactory() {
				return context.getMatrixFactory();
			}

			@Override
			public boolean isTrainingContext() {
				return context.isTrainingContext();
			}};
	}

	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.ACTIVATION_FUNCTION;
	}

	@Override
	public Neurons getInputNeurons() {
		return neurons;
	}

	@Override
	public Neurons getOutputNeurons() {
		return neurons;
	}
	
}
