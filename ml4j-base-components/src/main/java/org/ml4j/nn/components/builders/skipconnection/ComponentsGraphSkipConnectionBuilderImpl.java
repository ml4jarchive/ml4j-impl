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

import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;

public class ComponentsGraphSkipConnectionBuilderImpl<P extends ComponentsContainer<Neurons>> extends ComponentsNestedGraphBuilderImpl<P, ComponentsGraphSkipConnectionBuilder<P>> implements ComponentsGraphSkipConnectionBuilder<P> {

	public ComponentsGraphSkipConnectionBuilderImpl(Supplier<P> parentGraph, DirectedComponentFactory directedComponentFactory, BaseGraphBuilderState builderState, 
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(parentGraph, directedComponentFactory, builderState, components);
	}

	@Override
	public P endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parentGraph.get();
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<P> getBuilder() {
		return this;
	}

	@Override
	protected ComponentsGraphSkipConnectionBuilder<P> createNewNestedGraphBuilder() {
		return new ComponentsGraphSkipConnectionBuilderImpl<>(parentGraph, directedComponentFactory, initialBuilderState, components);
	}
}
