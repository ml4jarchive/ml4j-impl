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
package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.builders.initial.InitialComponents3DGraphBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinition;
import org.ml4j.nn.definitions.Component3DtoNon3DGraphDefinition;

public class Components3DSubGraphBuilderImpl<P extends Components3DGraphBuilder<P, Q, T>, Q extends ComponentsGraphBuilder<Q, T>, T extends NeuralComponent>
		extends ComponentsNested3DGraphBuilderImpl<P, Components3DSubGraphBuilder<P, Q, T>, ComponentsSubGraphBuilder<Q, T>, T>
		implements Components3DSubGraphBuilder<P, Q, T>, PathEnder<P, Components3DSubGraphBuilder<P, Q, T>> {

	private Supplier<P> previousSupplier;
	private ComponentsSubGraphBuilder<Q, T> builder;
	
	public Components3DSubGraphBuilderImpl(Supplier<P> previousSupplier, NeuralComponentFactory<T> directedComponentFactory,
			Base3DGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		super(previousSupplier, directedComponentFactory, builderState, directedComponentsContext, components);
		this.previousSupplier = previousSupplier;
	}

	@Override
	public Components3DSubGraphBuilder<P, Q, T> get3DBuilder() {
		return this;
	}

	@Override
	public ComponentsSubGraphBuilder<Q, T> getBuilder() {
		if (builder != null) {
			return builder;
		} else {
			previousSupplier.get().addComponents(getComponents());
			builder =  new ComponentsSubGraphBuilderImpl<>(() -> previousSupplier.get().getBuilder(), directedComponentFactory,
					builderState.getNon3DBuilderState(), directedComponentsContext, new ArrayList<>());
			return builder;
		}
	}
	
	@Override
	public Components3DSubGraphBuilder<P, Q, T> withPath() {
		completeNestedGraph(false);
		return createNewNestedGraphBuilder();
	}

	@Override
	public Components3DSubGraphBuilder<P, Q, T> withActivationFunction(
			DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return this;
	}

	@Override
	protected Components3DSubGraphBuilder<P, Q, T> createNewNestedGraphBuilder() {
		// FIX
		return new Components3DSubGraphBuilderImpl<>(previousSupplier,
				directedComponentFactory, initialBuilderState, directedComponentsContext, new ArrayList<>());
	}

	@Override
	public PathEnder<P, Components3DSubGraphBuilder<P, Q, T>> endPath() {
		completeNestedGraph(false);
		return this;
	}

	@Override
	public Components3DGraphBuilder<Components3DSubGraphBuilder<P, Q, T>, ComponentsSubGraphBuilder<Q, T>, T> withComponents(
			Components3DGraphBuilder<?, ?, T> builder) {
		addComponents(builder.getComponents());
		return this;
	}
	
	@Override
	public Components3DGraphBuilder<Components3DSubGraphBuilder<P, Q, T>, ComponentsSubGraphBuilder<Q, T>, T> withComponent(
			T component) {
		addComponents(Arrays.asList(component));
		return this;
	}

	@Override
	public Components3DGraphBuilder<Components3DSubGraphBuilder<P, Q, T>, ComponentsSubGraphBuilder<Q, T>, T> withComponents(
			List<T> components) {
		addComponents(components);
		return this;
	}

	@Override
	public Components3DGraphBuilder<Components3DSubGraphBuilder<P, Q, T>, ComponentsSubGraphBuilder<Q, T>, T> withComponentDefinition(
			Component3Dto3DGraphDefinition componentDefinition) {
		InitialComponents3DGraphBuilder<T> builder = new InitialComponents3DGraphBuilderImpl<T>(directedComponentFactory, directedComponentsContext, componentDefinition.getInputNeurons());
		addComponents(componentDefinition.createComponentGraph(builder).getComponents());
		return this;
	}

	@Override
	public Components3DGraphBuilder<Components3DSubGraphBuilder<P, Q, T>, ComponentsSubGraphBuilder<Q, T>, T> withComponentDefinition(
			List<Component3Dto3DGraphDefinition> componentDefinitions) {
		addComponents(componentDefinitions.stream().flatMap(d -> d.createComponentGraph(new InitialComponents3DGraphBuilderImpl<T>(directedComponentFactory, directedComponentsContext, d.getInputNeurons())).getComponents().stream()).collect(Collectors.toList()));
		return this;
	}
	
	@Override
	public ComponentsGraphBuilder<ComponentsSubGraphBuilder<Q, T>, T> withComponentDefinition(Component3DtoNon3DGraphDefinition componentDefinition) {
		InitialComponents3DGraphBuilder<T> builder = new InitialComponents3DGraphBuilderImpl<T>(directedComponentFactory, directedComponentsContext, componentDefinition.getInputNeurons());
		addComponents(componentDefinition.createComponentGraph(builder).getComponents());
		return getBuilder();
	}
	
}

