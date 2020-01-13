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
package org.ml4j.nn.components.builders.skipconnection;

import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.neurons.Neurons;

public class ComponentsGraphSkipConnectionBuilderImpl<P extends ComponentsContainer<Neurons, T>, T extends NeuralComponent> extends ComponentsNestedGraphBuilderImpl<P, ComponentsGraphSkipConnectionBuilder<P, T>, T> implements ComponentsGraphSkipConnectionBuilder<P, T> {

	public ComponentsGraphSkipConnectionBuilderImpl(Supplier<P> parentGraph, NeuralComponentFactory<T> directedComponentFactory, BaseGraphBuilderState builderState, 
			DirectedComponentsContext directedComponentsContext, List<T> components) {
		super(parentGraph, directedComponentFactory, builderState, directedComponentsContext, components);
	}

	@Override
	public P endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parentGraph.get();
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<P, T> getBuilder() {
		return this;
	}

	@Override
	protected ComponentsGraphSkipConnectionBuilder<P, T> createNewNestedGraphBuilder() {
		return new ComponentsGraphSkipConnectionBuilderImpl<>(parentGraph, directedComponentFactory, initialBuilderState, directedComponentsContext, components);
	}
}
