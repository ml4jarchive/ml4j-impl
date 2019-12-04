package org.ml4j.nn.components.builders.axonsgraph;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.AxonsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class AxonsGraphBuilderImpl<C extends AxonsBuilder> extends BaseGraphBuilderImpl<C> implements AxonsGraphBuilder<C> {

	protected Supplier<C> previousSupplier;
	protected C builder;
	
	public AxonsGraphBuilderImpl(Supplier<C> previousSupplier, DirectedAxonsComponentFactory directedAxonsComponentFactory,
			BaseGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(directedAxonsComponentFactory, builderState, components);
		this.previousSupplier = previousSupplier;
	}

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<C>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedAxonsComponentFactory, previousSupplier);
	}
	
	@Override
	public AxonsGraphSkipConnectionBuilder<C> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(previousSupplier, directedAxonsComponentFactory, builderState, new ArrayList<>());
	}
}
