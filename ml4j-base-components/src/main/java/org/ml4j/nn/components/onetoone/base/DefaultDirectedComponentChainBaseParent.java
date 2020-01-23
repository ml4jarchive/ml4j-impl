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
package org.ml4j.nn.components.onetoone.base;

import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuronsActivationComponent;
import org.ml4j.nn.components.generic.DirectedComponentChain;
import org.ml4j.nn.components.generic.DirectedComponentChainActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for implementations of DefaultDirectedComponentChain.
 * 
 * Encapsulates a sequential chain of DefaultChainableDirectedComponents
 * 
 * @author Michael Lavelle
 */
public abstract class DefaultDirectedComponentChainBaseParent<L extends DefaultChainableDirectedComponent<? extends A, ?>, 
	A extends DefaultChainableDirectedComponentActivation, CH extends DirectedComponentChainActivation<NeuronsActivation, A>> implements 
	DirectedComponentChain<NeuronsActivation, L, A, CH>, NeuronsActivationComponent {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedComponentChainBaseParent.class);
	
	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;
	
	protected List<L> sequentialComponents;

	public DefaultDirectedComponentChainBaseParent(List<L> sequentialComponents) {
		this.sequentialComponents = sequentialComponents;
	}

	protected <X, Y> Y forwardPropagate(NeuronsActivation input, DefaultChainableDirectedComponent<? extends Y, X> component, DirectedComponentsContext context) {
		return component.forwardPropagate(input, component.getContext(context));
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext context) {
		return context;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return sequentialComponents.stream().flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
	}

	@Override
	public List<L> getComponents() {
		return sequentialComponents;
	}

	public Neurons getInputNeurons() {
		return sequentialComponents.get(0).getInputNeurons();
	}

	public Neurons getOutputNeurons() {
		return sequentialComponents.get(sequentialComponents.size() - 1).getOutputNeurons();
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return sequentialComponents.stream().map(c -> c.isSupported(format)).allMatch(Predicate.isEqual(true));
	}
	
	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return NeuronsActivationFormat.intersectOptionals(sequentialComponents.stream().map(c -> c.optimisedFor()).collect(Collectors.toList()));
	}
}
