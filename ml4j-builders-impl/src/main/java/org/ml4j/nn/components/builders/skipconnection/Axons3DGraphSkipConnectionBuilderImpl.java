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

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.base.BaseNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Axons3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;

public class Axons3DGraphSkipConnectionBuilderImpl<P extends Axons3DGraphBuilder<P, Q, T>, Q extends AxonsGraphBuilder<Q, T>, T extends NeuralComponent>
		extends BaseNested3DGraphBuilderImpl<P, Axons3DGraphSkipConnectionBuilder<P, Q, T>, AxonsGraphSkipConnectionBuilder<Q, T>, T>
		implements Axons3DGraphSkipConnectionBuilder<P, Q, T>, PathEnder<P, Axons3DGraphSkipConnectionBuilder<P, Q, T>> {

	private AxonsGraphSkipConnectionBuilder<Q, T> builder;
	private Supplier<Q> parentNon3DGraph;

	
	public Axons3DGraphSkipConnectionBuilderImpl(Supplier<P> parent3DGraph, Supplier<Q> parentNon3DGraph, NeuralComponentFactory<T> directedComponentFactory,
			Base3DGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		super(parent3DGraph, directedComponentFactory, builderState, directedComponentsContext, components);
		this.parentNon3DGraph = parentNon3DGraph;
	}
	

	@Override
	public P endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parent3DGraph.get();
	}
	
	@Override
	public Axons3DGraphSkipConnectionBuilder<P, Q, T> withPath() {
		completeNestedGraph(false);
		return createNewNestedGraphBuilder();
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<P, Q, T> get3DBuilder() {
		return this;
	}

	@Override
	public AxonsGraphSkipConnectionBuilder<Q, T> getBuilder() {
		if (builder != null) {
			return builder;
		} else {
			parent3DGraph.get().addComponents(this.getComponents()); 
			builder =  new AxonsGraphSkipConnectionBuilderImpl<>(parentNon3DGraph, directedComponentFactory,
					builderState.getNon3DBuilderState(), directedComponentsContext, new ArrayList<>());
			return builder;
		}
	}
	
	@Override
	public ParallelPathsBuilder<Axons3DSubGraphBuilder<Axons3DGraphSkipConnectionBuilder<P, Q, T>, AxonsGraphSkipConnectionBuilder<Q, T>, T>> withParallelPaths() {
		return new Axons3DParallelPathsBuilderImpl<>(directedComponentFactory, this::get3DBuilder, this::getBuilder, directedComponentsContext);
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<Axons3DGraphSkipConnectionBuilder<P, Q, T>, AxonsGraphSkipConnectionBuilder<Q, T>, T> withSkipConnection() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory, builderState, directedComponentsContext, new ArrayList<>());
	}

	@Override
	protected Axons3DGraphSkipConnectionBuilder<P, Q, T> createNewNestedGraphBuilder() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(parent3DGraph, parentNon3DGraph, directedComponentFactory, initialBuilderState, directedComponentsContext, new ArrayList<>());
	}
}
