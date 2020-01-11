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
package org.ml4j.nn.components.builders.skipconnection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.builders.initial.InitialComponents3DGraphBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.definitions.Components3DGraphDefinition;

public class Components3DGraphSkipConnectionBuilderImpl<P extends Components3DGraphBuilder<P, Q, T>, Q extends ComponentsGraphBuilder<Q, T>, T extends NeuralComponent>
		extends ComponentsNested3DGraphBuilderImpl<P, Components3DGraphSkipConnectionBuilder<P, Q, T>, ComponentsGraphSkipConnectionBuilder<Q, T>, T>
		implements Components3DGraphSkipConnectionBuilder<P, Q, T>, PathEnder<P, Components3DGraphSkipConnectionBuilder<P, Q, T>>{

	private ComponentsGraphSkipConnectionBuilder<Q, T> builder;
	
	public Components3DGraphSkipConnectionBuilderImpl(Supplier<P> parentGraph, NeuralComponentFactory<T> directedComponentFactory,
			Base3DGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		super(parentGraph, directedComponentFactory, builderState, directedComponentsContext, components);
	}
	

	@Override
	public P endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parent3DGraph.get();
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q, T> get3DBuilder() {
		return this;
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<Q, T> getBuilder() {
		if (builder != null) {
			return builder;
		} else {
			builder =  new ComponentsGraphSkipConnectionBuilderImpl<>(() -> parent3DGraph.get().getBuilder(), directedComponentFactory,
					builderState.getNon3DBuilderState(), directedComponentsContext, getComponents());
			return builder;
		}
	}

	@Override
	protected Components3DGraphSkipConnectionBuilder<P, Q, T> createNewNestedGraphBuilder() {
		return new Components3DGraphSkipConnectionBuilderImpl<>(parent3DGraph, directedComponentFactory, initialBuilderState, directedComponentsContext, new ArrayList<>());
	}
	
	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q, T> withPath() {
		completeNestedGraph(false);
		return createNewNestedGraphBuilder();
	}


	@Override
	public Components3DGraphBuilder<Components3DGraphSkipConnectionBuilder<P, Q, T>, ComponentsGraphSkipConnectionBuilder<Q, T>, T> withComponents(
			Components3DGraphBuilder<?, ?, T> builder) {
		addComponents(builder.getComponents());
		return this;
	}
	

	@Override
	public Components3DGraphBuilder<Components3DGraphSkipConnectionBuilder<P, Q, T>, ComponentsGraphSkipConnectionBuilder<Q, T>, T> withComponent(
			T component) {
		addComponents(Arrays.asList(component));
		return this;
	}


	@Override
	public Components3DGraphBuilder<Components3DGraphSkipConnectionBuilder<P, Q, T>, ComponentsGraphSkipConnectionBuilder<Q, T>, T> withComponents(
			List<T> components) {
		addComponents(components);
		return this;
	}


	@Override
	public Components3DGraphBuilder<Components3DGraphSkipConnectionBuilder<P, Q, T>, ComponentsGraphSkipConnectionBuilder<Q, T>, T> withComponentDefinition(
			Components3DGraphDefinition componentDefinition) {
		InitialComponents3DGraphBuilder<T> builder = new InitialComponents3DGraphBuilderImpl<T>(directedComponentFactory, directedComponentsContext, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		addComponents(componentDefinition.createComponentGraph(builder).getComponents());
		return this;
	}


	@Override
	public Components3DGraphBuilder<Components3DGraphSkipConnectionBuilder<P, Q, T>, ComponentsGraphSkipConnectionBuilder<Q, T>, T> withComponentDefinition(
			List<Components3DGraphDefinition> componentDefinitions) {
		InitialComponents3DGraphBuilder<T> builder = new InitialComponents3DGraphBuilderImpl<T>(directedComponentFactory, directedComponentsContext, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		addComponents(componentDefinitions.stream().flatMap(d -> d.createComponentGraph(builder).getComponents().stream()).collect(Collectors.toList()));
		return this;
	}
}
