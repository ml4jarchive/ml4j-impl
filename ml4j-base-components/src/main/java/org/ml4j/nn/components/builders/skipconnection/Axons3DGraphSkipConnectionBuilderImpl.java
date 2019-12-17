package org.ml4j.nn.components.builders.skipconnection;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.PathCombinationStrategy;
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
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class Axons3DGraphSkipConnectionBuilderImpl<P extends Axons3DGraphBuilder<P, Q>, Q extends AxonsGraphBuilder<Q>>
		extends BaseNested3DGraphBuilderImpl<P, Axons3DGraphSkipConnectionBuilder<P, Q>, AxonsGraphSkipConnectionBuilder<Q>>
		implements Axons3DGraphSkipConnectionBuilder<P, Q>, PathEnder<P, Axons3DGraphSkipConnectionBuilder<P, Q>> {

	private AxonsGraphSkipConnectionBuilder<Q> builder;
	private Supplier<Q> parentNon3DGraph;

	
	public Axons3DGraphSkipConnectionBuilderImpl(Supplier<P> parent3DGraph, Supplier<Q> parentNon3DGraph, DirectedComponentFactory directedComponentFactory,
			Base3DGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(parent3DGraph, directedComponentFactory, builderState, components);
		this.parentNon3DGraph = parentNon3DGraph;
	}
	

	@Override
	public P endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parent3DGraph.get();
	}
	
	@Override
	public Axons3DGraphSkipConnectionBuilder<P, Q> withPath() {
		completeNestedGraph(false);
		return createNewNestedGraphBuilder();
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<P, Q> get3DBuilder() {
		return this;
	}

	@Override
	public AxonsGraphSkipConnectionBuilder<Q> getBuilder() {
		if (builder != null) {
			return builder;
		} else {
			parent3DGraph.get().addComponents(this.getComponents()); 
			builder =  new AxonsGraphSkipConnectionBuilderImpl<>(parentNon3DGraph, directedComponentFactory,
					builderState.getNon3DBuilderState(), new ArrayList<>());
			return builder;
		}
	}
	
	@Override
	public ParallelPathsBuilder<Axons3DSubGraphBuilder<Axons3DGraphSkipConnectionBuilder<P, Q>, AxonsGraphSkipConnectionBuilder<Q>>> withParallelPaths() {
		return new Axons3DParallelPathsBuilderImpl<>(directedComponentFactory, this::get3DBuilder, this::getBuilder);
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<Axons3DGraphSkipConnectionBuilder<P, Q>, AxonsGraphSkipConnectionBuilder<Q>> withSkipConnection() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	protected Axons3DGraphSkipConnectionBuilder<P, Q> createNewNestedGraphBuilder() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(parent3DGraph, parentNon3DGraph, directedComponentFactory, initialBuilderState, new ArrayList<>());
	}
}
