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
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinition;
import org.ml4j.nn.definitions.Component3DtoNon3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons3D;

public class Components3DGraphSkipConnectionBuilderImpl<P extends Components3DGraphBuilder<P, Q, T>, Q extends ComponentsGraphBuilder<Q, T>, T extends NeuralComponent<?>>
		extends
		ComponentsNested3DGraphBuilderImpl<P, Components3DGraphSkipConnectionBuilder<P, Q, T>, ComponentsGraphSkipConnectionBuilder<Q, T>, T>
		implements Components3DGraphSkipConnectionBuilder<P, Q, T>,
		PathEnder<P, Components3DGraphSkipConnectionBuilder<P, Q, T>> {

	private ComponentsGraphSkipConnectionBuilder<Q, T> builder;

	public Components3DGraphSkipConnectionBuilderImpl(Supplier<P> parentGraph,
			NeuralComponentFactory<T> directedComponentFactory, Base3DGraphBuilderState builderState,
			List<T> components) {
		super(parentGraph, directedComponentFactory, builderState, components);
	}

	@Override
	public P endSkipConnection(String name) {
		completeNestedGraph(true);
		completeNestedGraphs(name, PathCombinationStrategy.ADDITION);
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
			builder = new ComponentsGraphSkipConnectionBuilderImpl<>(() -> parent3DGraph.get().getBuilder(),
					directedComponentFactory, builderState.getNon3DBuilderState(),
					getComponents());
			return builder;
		}
	}

	@Override
	protected Components3DGraphSkipConnectionBuilder<P, Q, T> createNewNestedGraphBuilder() {
		return new Components3DGraphSkipConnectionBuilderImpl<>(parent3DGraph, directedComponentFactory,
				initialBuilderState, new ArrayList<>());
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q, T> withPath() {
		completeNestedGraph(false);
		return createNewNestedGraphBuilder();
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q, T> withComponentDefinition(
			Component3Dto3DGraphDefinition componentDefinition) {
		addAxonsIfApplicable();
		InitialComponents3DGraphBuilder<T> builder = new InitialComponents3DGraphBuilderImpl<T>(
				directedComponentFactory, componentDefinition.getInputNeurons());
		addComponents(componentDefinition.createComponentGraph(builder, directedComponentFactory).getComponents());
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setCurrentNeurons(componentDefinition.getOutputNeurons());
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		return this;
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<Q, T> withComponentDefinition(
			Component3DtoNon3DGraphDefinition componentDefinition) {
		addAxonsIfApplicable();
		InitialComponents3DGraphBuilder<T> builder = new InitialComponents3DGraphBuilderImpl<T>(
				directedComponentFactory, componentDefinition.getInputNeurons());
		addComponents(componentDefinition.createComponentGraph(builder, directedComponentFactory).getComponents());
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		ComponentsGraphSkipConnectionBuilder<Q, T> nextBuilder = getBuilder();
		nextBuilder.getComponentsGraphNeurons().setCurrentNeurons(componentDefinition.getOutputNeurons());
		return nextBuilder;
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q, T> with3DComponent(T component, Neurons3D endNeurons) {
		addAxonsIfApplicable();
		addComponents(Arrays.asList(component));
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setCurrentNeurons(endNeurons);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		return this;
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<Q, T> withNon3DComponent(T component) {
		addAxonsIfApplicable();
		addComponents(Arrays.asList(component));
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		ComponentsGraphSkipConnectionBuilder<Q, T> nextBuilder = getBuilder();
		nextBuilder.getComponentsGraphNeurons().setCurrentNeurons(component.getOutputNeurons());
		return nextBuilder;
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<Q, T> withComponents(ComponentsGraphBuilder<?, T> builder) {
		addAxonsIfApplicable();
		List<T> componentsToAdd = builder.getComponents();
		addComponents(componentsToAdd);
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		ComponentsGraphSkipConnectionBuilder<Q, T> nextBuilder = getBuilder();
		nextBuilder.getComponentsGraphNeurons()
				.setCurrentNeurons(componentsToAdd.get(componentsToAdd.size() - 1).getOutputNeurons());
		return nextBuilder;
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q, T> withComponents(Components3DGraphBuilder<?, ?, T> builder,
			Neurons3D endNeurons) {
		addAxonsIfApplicable();
		List<T> componentsToAdd = builder.getComponents();
		addComponents(componentsToAdd);
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		return this;
	}

}
