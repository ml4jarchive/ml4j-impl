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
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilder;
import org.ml4j.nn.components.builders.base.BaseNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;

public class AxonsGraphSkipConnectionBuilderImpl<C extends ComponentsContainer<Neurons>> extends BaseNestedGraphBuilderImpl<C, AxonsGraphSkipConnectionBuilder<C>> 
implements AxonsGraphSkipConnectionBuilder<C>, SkipConnectionEnder<C> {

	protected C builder;
	
	public AxonsGraphSkipConnectionBuilderImpl(Supplier<C> previousSupplier, DirectedComponentFactory directedComponentFactory,
			BaseGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(previousSupplier, directedComponentFactory, builderState, components);
	}

	@Override
	public AxonsGraphSkipConnectionBuilder<C> getBuilder() {
		return this;
	}


	@Override
	public AxonsGraphSkipConnectionBuilder<AxonsGraphSkipConnectionBuilder<C>> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}
	

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<AxonsGraphSkipConnectionBuilder<C>>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedComponentFactory,this::getBuilder);
	}

	@Override
	public C endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parentGraph.get();
	}

	@Override
	protected AxonsGraphSkipConnectionBuilder<C> createNewNestedGraphBuilder() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(parentGraph, directedComponentFactory, initialBuilderState, new ArrayList<>());
	}
}
