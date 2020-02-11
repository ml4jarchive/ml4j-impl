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

import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.base.BaseNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.Axons3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

public class CompletedSynapsesAxons3DGraphBuilderImpl<P extends Axons3DBuilder<T>, Q extends AxonsBuilder<T>, T extends NeuralComponent<?>>
		extends
		BaseNested3DGraphBuilderImpl<P, CompletedSynapsesAxons3DGraphBuilder<P, Q, T>, CompletedSynapsesAxonsGraphBuilder<Q, T>, T>
		implements CompletedSynapsesAxons3DGraphBuilder<P, Q, T>, SynapsesEnder<P>,
		ParallelPathsBuilder<Axons3DSubGraphBuilder<CompletedSynapsesAxons3DGraphBuilder<P, Q, T>, CompletedSynapsesAxonsGraphBuilder<Q, T>, T>> {

	private Supplier<Q> parentNon3DGraph;
	private CompletedSynapsesAxonsGraphBuilderImpl<Q, T> builder;

	public CompletedSynapsesAxons3DGraphBuilderImpl(Supplier<P> parent3DGraph, Supplier<Q> parentNon3DGraph,
			NeuralComponentFactory<T> directedComponentFactory, Base3DGraphBuilderState builderState,
			DirectedComponentsContext directedComponentsContext, List<T> components) {
		super(parent3DGraph, directedComponentFactory, builderState, directedComponentsContext, components);
		this.parentNon3DGraph = parentNon3DGraph;
	}

	@Override
	public ParallelPathsBuilder<Axons3DSubGraphBuilder<CompletedSynapsesAxons3DGraphBuilder<P, Q, T>, CompletedSynapsesAxonsGraphBuilder<Q, T>, T>> withParallelPaths() {
		return this;
	}

	@Override
	public SynapsesEnder<P> withActivationFunction(String name, DifferentiableActivationFunction activationFunction) {
		addActivationFunction(name, activationFunction);
		return this;
	}

	@Override
	public SynapsesEnder<P> withActivationFunction(String name,ActivationFunctionType activationFunctionType, ActivationFunctionProperties activationFunctionProperties) {
		addActivationFunction(name, activationFunctionType, activationFunctionProperties);
		return this;
	}

	@Override
	public P endSynapses() {
		addAxonsIfApplicable();
		this.parent3DGraph.get().addAxonsIfApplicable();
		this.parent3DGraph.get().getComponentsGraphNeurons()
				.setCurrentNeurons(getComponentsGraphNeurons().getCurrentNeurons());
		this.parent3DGraph.get().getComponentsGraphNeurons()
				.setRightNeurons(getComponentsGraphNeurons().getRightNeurons());
		this.parent3DGraph.get().getComponentsGraphNeurons().setHasBiasUnit(getComponentsGraphNeurons().hasBiasUnit());
		// TODO ML Here we would add synapses instead of the chain
		T chain = directedComponentFactory.createDirectedComponentChain(getComponents());
		this.parent3DGraph.get().addComponent(chain);
		return parent3DGraph.get();
	}

	@Override
	public CompletedSynapsesAxons3DGraphBuilder<P, Q, T> get3DBuilder() {
		return this;
	}

	@Override
	public CompletedSynapsesAxonsGraphBuilder<Q, T> getBuilder() {
		if (builder == null) {
			addAxonsIfApplicable();
			builder = new CompletedSynapsesAxonsGraphBuilderImpl<>(parentNon3DGraph, directedComponentFactory,
					builderState.getNon3DBuilderState(), directedComponentsContext, getComponents());
		}
		return builder;
	}

	@Override
	public Axons3DSubGraphBuilder<CompletedSynapsesAxons3DGraphBuilder<P, Q, T>, CompletedSynapsesAxonsGraphBuilder<Q, T>, T> withPath() {
		return new Axons3DSubGraphBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory,
				builderState, directedComponentsContext, new ArrayList<>());
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<CompletedSynapsesAxons3DGraphBuilder<P, Q, T>, CompletedSynapsesAxonsGraphBuilder<Q, T>, T> withSkipConnection() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, this::getBuilder,
				directedComponentFactory, builderState, directedComponentsContext, new ArrayList<>());
	}

	@Override
	protected CompletedSynapsesAxons3DGraphBuilder<P, Q, T> createNewNestedGraphBuilder() {
		return new CompletedSynapsesAxons3DGraphBuilderImpl<>(parent3DGraph, parentNon3DGraph, directedComponentFactory,
				initialBuilderState, directedComponentsContext, new ArrayList<>());
	}
}
