package org.ml4j.nn.components.builders.skipconnection;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilder;
import org.ml4j.nn.components.builders.base.BaseNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class AxonsGraphSkipConnectionBuilderImpl<C extends ComponentsContainer<Neurons>> extends BaseNestedGraphBuilderImpl<C, AxonsGraphSkipConnectionBuilder<C>> 
implements AxonsGraphSkipConnectionBuilder<C>, SkipConnectionEnder<C> {

	protected C builder;
	
	public AxonsGraphSkipConnectionBuilderImpl(Supplier<C> previousSupplier, DirectedAxonsComponentFactory directedAxonsComponentFactory,
			BaseGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(previousSupplier, directedAxonsComponentFactory, builderState, components);
	}

	@Override
	public AxonsGraphSkipConnectionBuilder<C> getBuilder() {
		return this;
	}


	@Override
	public AxonsGraphSkipConnectionBuilder<AxonsGraphSkipConnectionBuilder<C>> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(this::getBuilder, directedAxonsComponentFactory, builderState, new ArrayList<>());
	}
	

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<AxonsGraphSkipConnectionBuilder<C>>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedAxonsComponentFactory,this::getBuilder);
	}

	@Override
	public C endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parentGraph.get();
	}

	@Override
	protected AxonsGraphSkipConnectionBuilder<C> createNewNestedGraphBuilder() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(parentGraph, directedAxonsComponentFactory, initialBuilderState, new ArrayList<>());
	}
}
