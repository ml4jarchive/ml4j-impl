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
package org.ml4j.nn.components.onetoone;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

public class TrailingActivationFunctionDirectedComponentChainImpl
		implements TrailingActivationFunctionDirectedComponentChain {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private List<DefaultChainableDirectedComponent<?, ?>> components;
	private DifferentiableActivationFunctionComponent finalDifferentiableActivationFunctionComponent;
	private DefaultDirectedComponentChain precedingChain;
	private DirectedComponentFactory directedComponentFactory;

	public DifferentiableActivationFunctionComponent getFinalComponent() {
		return finalDifferentiableActivationFunctionComponent;
	}

	public TrailingActivationFunctionDirectedComponentChainImpl(DirectedComponentFactory directedComponentFactory,
			List<? extends DefaultChainableDirectedComponent<?, ?>> components) {
		this.components = new ArrayList<>();
		this.components.addAll(components);
		this.directedComponentFactory = directedComponentFactory;
		List<DefaultChainableDirectedComponent<?, ?>> decomposedList = new ArrayList<>();
		for (DefaultChainableDirectedComponent<?, ?> component : decompose()) {
			decomposedList.add(component);
		}
		if (components.isEmpty()) {
			throw new IllegalArgumentException("Component list must contain at least one component");
		} else {
			ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?> finalComponent = decomposedList
					.get(decomposedList.size() - 1);
			if (NeuralComponentBaseType.ACTIVATION_FUNCTION.equals(finalComponent.getComponentType().getBaseType())) {
				finalDifferentiableActivationFunctionComponent = (DifferentiableActivationFunctionComponent) finalComponent;
				decomposedList.remove(decomposedList.size() - 1);
				this.precedingChain = directedComponentFactory.createDirectedComponentChain(decomposedList);

			} else {
				throw new IllegalArgumentException(
						"Decomposed component list must end with a differentiable activation function component");
			}
		}

	}

	private TrailingActivationFunctionDirectedComponentChainImpl(
			List<DefaultChainableDirectedComponent<?, ?>> components,
			DifferentiableActivationFunctionComponent finalDifferentiableActivationFunctionComponent,
			DefaultDirectedComponentChain precedingChain) {
		this.components = components;
		this.finalDifferentiableActivationFunctionComponent = finalDifferentiableActivationFunctionComponent;
		this.precedingChain = precedingChain;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		return components;
	}

	@Override
	public TrailingActivationFunctionDirectedComponentChainActivation forwardPropagate(NeuronsActivation input,
			DirectedComponentsContext context) {

		DefaultDirectedComponentChainActivation precedingChainActivation = precedingChain.forwardPropagate(input,
				context);
		DifferentiableActivationFunctionComponentActivation activationFunctionComponentActivation = finalDifferentiableActivationFunctionComponent
				.forwardPropagate(precedingChainActivation.getOutput(),
						context);
		// activationFunctionActivation.getInput().close();
		return new TrailingActivationFunctionDirectedComponentChainActivationImpl(this, precedingChainActivation,
				activationFunctionComponentActivation);
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext) {
		return directedComponentsContext;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		List<DefaultChainableDirectedComponent<?, ?>> allComponents = new ArrayList<>();
		allComponents.addAll(components);
		return directedComponentFactory.createDirectedComponentChain(allComponents).decompose();
	}

	@Override
	public TrailingActivationFunctionDirectedComponentChain dup() {

		List<DefaultChainableDirectedComponent<?, ?>> dupComponents = components.stream().map(c -> c.dup())
				.collect(Collectors.toList());

		return new TrailingActivationFunctionDirectedComponentChainImpl(dupComponents,
				finalDifferentiableActivationFunctionComponent.dup(), precedingChain.dup());
	}

	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.createSubType(
				NeuralComponentType.getBaseType(NeuralComponentBaseType.COMPONENT_CHAIN),
				"TRAILING_ACTIVATION_FUNCTION");
	}

	@Override
	public Neurons getInputNeurons() {
		return precedingChain.getInputNeurons();
	}

	@Override
	public Neurons getOutputNeurons() {
		return finalDifferentiableActivationFunctionComponent.getOutputNeurons();
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return precedingChain.isSupported(format) && finalDifferentiableActivationFunctionComponent.isSupported(format);
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return NeuronsActivationFormat.intersectOptionals(precedingChain.optimisedFor(), finalDifferentiableActivationFunctionComponent.optimisedFor());
	}

	@Override
	public String getName() {
		return precedingChain.getName() + ":" + finalDifferentiableActivationFunctionComponent.getName();
	}
}
