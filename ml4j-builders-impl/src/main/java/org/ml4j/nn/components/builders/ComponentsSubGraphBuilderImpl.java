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
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.neurons.Neurons;

public class ComponentsSubGraphBuilderImpl<P extends ComponentsContainer<Neurons, T>, T extends NeuralComponent>
		extends ComponentsNestedGraphBuilderImpl<P, ComponentsSubGraphBuilder<P, T>, T>
		implements ComponentsSubGraphBuilder<P, T>, PathEnder<P, ComponentsSubGraphBuilder<P, T>> {

	public ComponentsSubGraphBuilderImpl(Supplier<P> parentGraph, NeuralComponentFactory<T> directedComponentFactory,
			BaseGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		super(parentGraph, directedComponentFactory, builderState, directedComponentsContext, components);
	}

	@Override
	public PathEnder<P, ComponentsSubGraphBuilder<P, T>> endPath() {
		return this;
	}

	@Override
	public ComponentsSubGraphBuilder<P, T> getBuilder() {
		return this;
	}

	@Override
	protected ComponentsSubGraphBuilder<P, T> createNewNestedGraphBuilder() {
		return new ComponentsSubGraphBuilderImpl<>(parentGraph, directedComponentFactory, initialBuilderState,
				directedComponentsContext, new ArrayList<>());
	}
}
