/*
 * Copyright 2020 the original author or authors.
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
package org.ml4j.nn.components;

import org.ml4j.nn.neurons.Neurons;

/**
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of Neurons on the LHS of this component.
 * @param <R> The type of Neurons on the RHS of this component.
 */
public class NeuralComponentAdapter<L extends Neurons, R extends Neurons> implements NeuralComponent {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private NeuralComponentType<?> neuralComponentType;
	private L inputNeurons;
	private R outputNeurons;
	private String name;
	
	public NeuralComponentAdapter(String name, NeuralComponentType<?> neuralComponentType, L inputNeurons, R outputNeurons) {
		this.inputNeurons = inputNeurons;
		this.outputNeurons = outputNeurons;
		this.neuralComponentType = neuralComponentType;
		this.name = name;
	}

	@Override
	public NeuralComponentType<?> getComponentType() {
		return neuralComponentType;
	}

	@Override
	public Neurons getInputNeurons() {
		return inputNeurons;
	}

	@Override
	public Neurons getOutputNeurons() {
		return outputNeurons;
	}

	@Override
	public String getName() {
		return name;
	}

}
