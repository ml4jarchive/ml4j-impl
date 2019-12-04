package org.ml4j.nn.components.builders.synapses;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilder;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.AxonsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public class SynapsesAxonsGraphBuilderImpl<C extends AxonsBuilder> extends BaseGraphBuilderImpl<CompletedSynapsesAxonsGraphBuilder<C>> implements SynapsesAxonsGraphBuilder<C> {

	private Supplier<C> previousSupplier;
	private CompletedSynapsesAxonsGraphBuilder<C> builder;
	
	public SynapsesAxonsGraphBuilderImpl(Supplier<C> previousSupplier, DirectedAxonsComponentFactory directedAxonsComponentFactory,
			BaseGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(directedAxonsComponentFactory, builderState, components);
		this.previousSupplier = previousSupplier;
	}

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<CompletedSynapsesAxonsGraphBuilder<C>>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedAxonsComponentFactory, this::getBuilder);
	}
	
	@Override
	public AxonsGraphSkipConnectionBuilder<CompletedSynapsesAxonsGraphBuilder<C>> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(this::getBuilder, directedAxonsComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public CompletedSynapsesAxonsGraphBuilder<C> getBuilder() {
		if (builder == null) {
			builder = new CompletedSynapsesAxonsGraphBuilderImpl<>(previousSupplier, directedAxonsComponentFactory, builderState, components);
		}
		return builder;
	}

}
