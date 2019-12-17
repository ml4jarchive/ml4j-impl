package org.ml4j.nn.components.builders.axonsgraph;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.base.BaseNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.skipconnection.AxonsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons;

public class AxonsSubGraphBuilderImpl<C extends ComponentsContainer<Neurons>> extends BaseNestedGraphBuilderImpl<C, AxonsSubGraphBuilder<C>> implements AxonsSubGraphBuilder<C>, 
	PathEnder<C, AxonsSubGraphBuilder<C>> {

	protected C builder;
	
	public AxonsSubGraphBuilderImpl(Supplier<C> previousSupplier, DirectedComponentFactory directedComponentFactory,
			BaseGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(previousSupplier, directedComponentFactory, builderState, components);
	}

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<AxonsSubGraphBuilder<C>>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedComponentFactory, this::getBuilder);
	}
	
	@Override
	public AxonsGraphSkipConnectionBuilder<AxonsSubGraphBuilder<C>> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(() -> this, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public AxonsSubGraphBuilder<C> getBuilder() {
		return this;
	}

	@Override
	public AxonsSubGraphBuilder<C> withPath() {
		return new AxonsSubGraphBuilderImpl<>(parentGraph, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public PathEnder<C, AxonsSubGraphBuilder<C>> endPath() {
		return this;
	}

	@Override
	protected AxonsSubGraphBuilder<C> createNewNestedGraphBuilder() {
		return new AxonsSubGraphBuilderImpl<>(parentGraph, directedComponentFactory, initialBuilderState, components);
	}
}
