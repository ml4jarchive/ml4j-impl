package org.ml4j.nn.components.builders.axonsgraph;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.base.BaseNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Axons3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.skipconnection.Axons3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;

public class Axons3DSubGraphBuilderImpl<P extends Axons3DGraphBuilder<P, Q>, Q extends AxonsGraphBuilder<Q>>
		extends BaseNested3DGraphBuilderImpl<P, Axons3DSubGraphBuilder<P, Q>, AxonsSubGraphBuilder<Q>>
		implements Axons3DSubGraphBuilder<P, Q>, PathEnder<P, Axons3DSubGraphBuilder<P, Q>> {

	private Supplier<Q> parentNon3DGraph;
	private AxonsSubGraphBuilder<Q> builder;
	private Axons3DSubGraphBuilder<P, Q> currentPath;

	public Axons3DSubGraphBuilderImpl(Supplier<P> parentGraph, Supplier<Q> parentNon3DGraph, DirectedComponentFactory directedComponentFactory, Base3DGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(parentGraph, directedComponentFactory, builderState, components);
		this.parentNon3DGraph = parentNon3DGraph;
	}


	@Override
	public PathEnder<P, Axons3DSubGraphBuilder<P, Q>> endPath() {
		completeNestedGraph(false);
		return this;
	}

	@Override
	public Axons3DSubGraphBuilder<P, Q> get3DBuilder() {
		return this;
	}

	@Override
	public AxonsSubGraphBuilder<Q> getBuilder() {
		if (builder != null) {
			return builder;
		} else {
			parent3DGraph.get().addComponents(this.getComponents());
			builder = new AxonsSubGraphBuilderImpl<>(parentNon3DGraph,  directedComponentFactory,
					builderState.getNon3DBuilderState(), new ArrayList<>());
			return builder;
		}
	}

	@Override
	public ParallelPathsBuilder<Axons3DSubGraphBuilder<Axons3DSubGraphBuilder<P, Q>, AxonsSubGraphBuilder<Q>>> withParallelPaths() {
		return new Axons3DParallelPathsBuilderImpl<>(directedComponentFactory, this::get3DBuilder, this::getBuilder);
	}


	@Override
	public Axons3DGraphSkipConnectionBuilder<Axons3DSubGraphBuilder<P, Q>, AxonsSubGraphBuilder<Q>> withSkipConnection() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}


	@Override
	protected Axons3DSubGraphBuilder<P, Q> createNewNestedGraphBuilder() {
		return new Axons3DSubGraphBuilderImpl<>(parent3DGraph,
				parentNon3DGraph, directedComponentFactory, initialBuilderState, new ArrayList<>());
	}
	
	@Override
	public Axons3DSubGraphBuilder<P, Q> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		completeNestedGraph(false);
		currentPath = createNewNestedGraphBuilder();
		return currentPath;
	}
}
