package org.ml4j.nn.components.builders.synapses;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilder;
import org.ml4j.nn.components.builders.base.BaseNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Axons3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.Axons3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;

public class SynapsesAxons3DGraphBuilderImpl<C extends Axons3DBuilder, D extends AxonsBuilder> extends BaseNested3DGraphBuilderImpl<C, CompletedSynapsesAxons3DGraphBuilder<C, D>, CompletedSynapsesAxonsGraphBuilder<D>> implements SynapsesAxons3DGraphBuilder<C, D> {

	private Supplier<D> parentNon3DGraph;

	private CompletedSynapsesAxons3DGraphBuilder<C, D> builder3D;
	private CompletedSynapsesAxonsGraphBuilder<D> builder;
	
	public SynapsesAxons3DGraphBuilderImpl(Supplier<C> parent3DGraph, Supplier<D> parentNon3DGraph, DirectedComponentFactory directedComponentFactory,
			Base3DGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(parent3DGraph, directedComponentFactory, builderState, components);
		this.parentNon3DGraph = parentNon3DGraph;
	}
	
	@Override
	public ParallelPathsBuilder<Axons3DSubGraphBuilder<CompletedSynapsesAxons3DGraphBuilder<C, D>, CompletedSynapsesAxonsGraphBuilder<D>>> withParallelPaths() {
		return new Axons3DParallelPathsBuilderImpl<>(directedComponentFactory, this::get3DBuilder, this::getBuilder);
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<CompletedSynapsesAxons3DGraphBuilder<C, D>, CompletedSynapsesAxonsGraphBuilder<D>> withSkipConnection() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public CompletedSynapsesAxons3DGraphBuilder<C, D> get3DBuilder() {
		if (builder3D == null) {
			this.addAxonsIfApplicable();
			this.builder3D =  new CompletedSynapsesAxons3DGraphBuilderImpl<>(parent3DGraph, parentNon3DGraph, directedComponentFactory, builderState, getComponents());
		}
		return builder3D;
	}

	@Override
	public CompletedSynapsesAxonsGraphBuilder<D> getBuilder() {
		addAxonsIfApplicable();
		if (builder == null) {
			builder = new CompletedSynapsesAxonsGraphBuilderImpl<>(parentNon3DGraph, directedComponentFactory, builderState.getNon3DBuilderState(), getComponents());
		}
		return builder;
	}

	@Override
	protected CompletedSynapsesAxons3DGraphBuilder<C, D> createNewNestedGraphBuilder() {
		return new CompletedSynapsesAxons3DGraphBuilderImpl<>(parent3DGraph, parentNon3DGraph, directedComponentFactory, initialBuilderState, new ArrayList<>());
	}
}
