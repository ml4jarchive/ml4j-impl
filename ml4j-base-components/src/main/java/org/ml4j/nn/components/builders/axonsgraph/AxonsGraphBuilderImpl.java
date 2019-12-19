package org.ml4j.nn.components.builders.axonsgraph;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.AxonsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;

public abstract class AxonsGraphBuilderImpl<C extends AxonsBuilder> extends BaseGraphBuilderImpl<C> implements AxonsGraphBuilder<C> {

	protected Supplier<C> previousSupplier;
	protected C builder;
	
	public AxonsGraphBuilderImpl(Supplier<C> previousSupplier, DirectedComponentFactory directedComponentFactory,
			BaseGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(directedComponentFactory, builderState, components);
		this.previousSupplier = previousSupplier;
	}

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<C>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedComponentFactory, previousSupplier);
	}
	
	@Override
	public AxonsGraphSkipConnectionBuilder<C> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(previousSupplier, directedComponentFactory, builderState, new ArrayList<>());
	}
}
