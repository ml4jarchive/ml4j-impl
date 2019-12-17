package org.ml4j.nn.components.builders.synapses;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.DefaultDirectedComponentChain;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.base.BaseNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.Axons3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class CompletedSynapsesAxons3DGraphBuilderImpl<P extends Axons3DBuilder, Q extends AxonsBuilder> extends BaseNested3DGraphBuilderImpl<P, CompletedSynapsesAxons3DGraphBuilder<P, Q>, CompletedSynapsesAxonsGraphBuilder<Q>> implements CompletedSynapsesAxons3DGraphBuilder<P, Q>, SynapsesEnder<P>, ParallelPathsBuilder<Axons3DSubGraphBuilder<CompletedSynapsesAxons3DGraphBuilder<P, Q>, CompletedSynapsesAxonsGraphBuilder<Q>>> {

	private Supplier<Q> parentNon3DGraph;	
	private CompletedSynapsesAxonsGraphBuilderImpl<Q> builder;
	
	public CompletedSynapsesAxons3DGraphBuilderImpl(Supplier<P> parent3DGraph, Supplier<Q> parentNon3DGraph, DirectedComponentFactory directedComponentFactory,
			Base3DGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(parent3DGraph, directedComponentFactory, builderState, components);
		this.parentNon3DGraph = parentNon3DGraph;
	}

	@Override
	public ParallelPathsBuilder<Axons3DSubGraphBuilder<CompletedSynapsesAxons3DGraphBuilder<P, Q>, CompletedSynapsesAxonsGraphBuilder<Q>>> withParallelPaths() {
		return this;
	}

	@Override
	public SynapsesEnder<P> withActivationFunction(
			DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return this;
	}
	
	@Override
	public P endSynapses() {
		addAxonsIfApplicable();
		this.parent3DGraph.get().addAxonsIfApplicable();
		this.parent3DGraph.get().getComponentsGraphNeurons().setCurrentNeurons(getComponentsGraphNeurons().getCurrentNeurons());
		this.parent3DGraph.get().getComponentsGraphNeurons().setRightNeurons(getComponentsGraphNeurons().getRightNeurons());
		this.parent3DGraph.get().getComponentsGraphNeurons().setHasBiasUnit(getComponentsGraphNeurons().hasBiasUnit());
		// TODO ML Here we would add synapses instead of the chain
		DefaultDirectedComponentChain
			chain = new DefaultDirectedComponentChainImpl(getComponents());
		this.parent3DGraph.get().addComponent(chain);
		return parent3DGraph.get();
	}

	@Override
	public CompletedSynapsesAxons3DGraphBuilder<P, Q> get3DBuilder() {
		return this;
	}

	@Override
	public CompletedSynapsesAxonsGraphBuilder<Q> getBuilder() {
		if (builder == null) {
			addAxonsIfApplicable();
			builder =  new CompletedSynapsesAxonsGraphBuilderImpl<>(parentNon3DGraph, directedComponentFactory, builderState.getNon3DBuilderState(), getComponents());
		}
		return builder;
	}

	@Override
	public Axons3DSubGraphBuilder<CompletedSynapsesAxons3DGraphBuilder<P, Q>, CompletedSynapsesAxonsGraphBuilder<Q>> withPath() {
		return new Axons3DSubGraphBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<CompletedSynapsesAxons3DGraphBuilder<P, Q>, CompletedSynapsesAxonsGraphBuilder<Q>> withSkipConnection() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	protected CompletedSynapsesAxons3DGraphBuilder<P, Q> createNewNestedGraphBuilder() {
		return new CompletedSynapsesAxons3DGraphBuilderImpl<>(parent3DGraph, parentNon3DGraph, directedComponentFactory, initialBuilderState, new ArrayList<>());
	}
}