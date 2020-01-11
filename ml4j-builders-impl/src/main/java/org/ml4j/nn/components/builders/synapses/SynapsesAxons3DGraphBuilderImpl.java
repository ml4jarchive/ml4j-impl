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
package org.ml4j.nn.components.builders.synapses;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilder;
import org.ml4j.nn.components.builders.base.BaseNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Axons3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.Axons3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

public class SynapsesAxons3DGraphBuilderImpl<C extends Axons3DBuilder<T>, D extends AxonsBuilder<T>, T extends NeuralComponent> extends BaseNested3DGraphBuilderImpl<C, CompletedSynapsesAxons3DGraphBuilder<C, D, T>, CompletedSynapsesAxonsGraphBuilder<D, T>, T> implements SynapsesAxons3DGraphBuilder<C, D, T> {

	private Supplier<D> parentNon3DGraph;

	private CompletedSynapsesAxons3DGraphBuilder<C, D, T> builder3D;
	private CompletedSynapsesAxonsGraphBuilder<D, T> builder;
	
	public SynapsesAxons3DGraphBuilderImpl(Supplier<C> parent3DGraph, Supplier<D> parentNon3DGraph, NeuralComponentFactory<T> directedComponentFactory,
			Base3DGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		super(parent3DGraph, directedComponentFactory, builderState, directedComponentsContext, components);
		this.parentNon3DGraph = parentNon3DGraph;
	}
	
	@Override
	public ParallelPathsBuilder<Axons3DSubGraphBuilder<CompletedSynapsesAxons3DGraphBuilder<C, D, T>, CompletedSynapsesAxonsGraphBuilder<D, T>, T>> withParallelPaths() {
		return new Axons3DParallelPathsBuilderImpl<>(directedComponentFactory, this::get3DBuilder, this::getBuilder, directedComponentsContext);
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<CompletedSynapsesAxons3DGraphBuilder<C, D, T>, CompletedSynapsesAxonsGraphBuilder<D, T>, T> withSkipConnection() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory, builderState, directedComponentsContext, new ArrayList<>());
	}

	@Override
	public CompletedSynapsesAxons3DGraphBuilder<C, D, T> get3DBuilder() {
		if (builder3D == null) {
			this.addAxonsIfApplicable();
			this.builder3D =  new CompletedSynapsesAxons3DGraphBuilderImpl<>(parent3DGraph, parentNon3DGraph, directedComponentFactory, builderState, directedComponentsContext, getComponents());
		}
		return builder3D;
	}

	@Override
	public CompletedSynapsesAxonsGraphBuilder<D, T> getBuilder() {
		addAxonsIfApplicable();
		if (builder == null) {
			builder = new CompletedSynapsesAxonsGraphBuilderImpl<>(parentNon3DGraph, directedComponentFactory, builderState.getNon3DBuilderState(), directedComponentsContext, getComponents());
		}
		return builder;
	}

	@Override
	protected CompletedSynapsesAxons3DGraphBuilder<C, D, T> createNewNestedGraphBuilder() {
		return new CompletedSynapsesAxons3DGraphBuilderImpl<>(parent3DGraph, parentNon3DGraph, directedComponentFactory, initialBuilderState, directedComponentsContext, new ArrayList<>());
	}
}
